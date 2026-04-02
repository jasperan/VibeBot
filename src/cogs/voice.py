import asyncio
import json
import logging

import discord
from discord import app_commands
from discord.ext import commands

import webrtcvad

from src.asr_client import ASRClient
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.voice_pipeline import VoicePipeline
from src.audio import resample_pcm, stereo_to_mono, PCMStreamSource

log = logging.getLogger("vibebot.voice")

VAD_FRAME_MS = 30
VAD_FRAME_BYTES = int(16000 * 2 * VAD_FRAME_MS / 1000)

PERSONALITY_PRESETS = {
    "default": "You are VibeBot, a friendly voice assistant in a Discord voice channel. Keep responses concise and conversational, 1-2 sentences max.",
    "pirate": "You are VibeBot, a pirate voice assistant. Respond like a swashbuckling sea captain. Keep it to 1-2 sentences, arrr!",
    "shakespeare": "You are VibeBot, speaking in Shakespearean English. Thou art eloquent yet brief. 1-2 sentences, forsooth.",
    "sarcastic": "You are VibeBot, dripping with dry sarcasm. Keep responses to 1-2 sentences.",
    "zen": "You are VibeBot, a calm zen master. Respond with quiet wisdom and tranquility. 1-2 sentences.",
    "hype": "You are VibeBot, an extremely enthusiastic hype person. Everything is AMAZING. 1-2 sentences, max energy!",
}

MUSIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Search YouTube and play a song in the voice channel. Use when the user asks to play music, a song, or audio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song name, artist, or search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_song",
            "description": "Skip the currently playing song and play the next one in queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Stop all music playback and clear the queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the music playback volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "number", "description": "Volume level from 0.0 (silent) to 1.0 (full)"}
                },
                "required": ["level"],
            },
        },
    },
]


class VoiceCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.is_listening = False
        self._listen_task: asyncio.Task | None = None
        self._is_responding = False
        self._response_task: asyncio.Task | None = None
        self._pending_actions: list = []
        self._current_personality = "default"

        vc = bot.config["voice"]
        lc = bot.config["llm"]

        self._asr = ASRClient(base_url=vc["asr_url"])
        self._llm = LLMClient(
            base_url=lc["base_url"],
            model=lc["model"],
            max_tokens=lc["max_tokens"],
            system_prompt=lc["system_prompt"],
            backend=lc.get("backend", "openai"),
        )
        self._tts = TTSClient(
            host=vc["tts_host"],
            port=vc["tts_port"],
            voice=vc["tts_voice"],
            cfg=vc["tts_cfg"],
            steps=vc["tts_steps"],
        )
        self._pipeline = VoicePipeline(
            asr_client=self._asr,
            llm_client=self._llm,
            tts_client=self._tts,
            context_window=vc["context_window"],
        )
        # Wire up voice-commanded music tools
        self._pipeline.tools = MUSIC_TOOLS
        self._pipeline.tool_executor = self._execute_tool

        self._vad = webrtcvad.Vad(vc["vad_aggressiveness"])
        self._silence_threshold_ms = vc["silence_threshold_ms"]
        self._max_utterance_bytes = int(vc["max_utterance_seconds"] * 16000 * 2)

    # ── Slash commands ──────────────────────────────────────────────

    @app_commands.command(name="join", description="Join your voice channel")
    async def join(self, interaction: discord.Interaction):
        if not interaction.user.voice:
            await interaction.response.send_message(
                "You need to be in a voice channel.", ephemeral=True
            )
            return
        channel = interaction.user.voice.channel
        if interaction.guild.voice_client:
            await interaction.guild.voice_client.move_to(channel)
        else:
            await channel.connect()
        await interaction.response.send_message(f"Joined {channel.name}", ephemeral=True)

    @app_commands.command(name="leave", description="Leave voice channel")
    async def leave(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc:
            await interaction.response.send_message("Not in a voice channel.", ephemeral=True)
            return
        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            self._response_task = None
        self._is_responding = False
        self._pipeline.clear_context()
        music_cog = self.bot.get_cog("MusicCog")
        if music_cog:
            music_cog._clear_state()
        await vc.disconnect()
        if self.bot.services.is_running:
            await self.bot.services.shutdown()
        await interaction.response.send_message("Left voice channel.", ephemeral=True)

    @app_commands.command(name="listen", description="Toggle voice conversation mode")
    async def listen(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc:
            await interaction.response.send_message(
                "Not in a voice channel. Use /join first.", ephemeral=True
            )
            return
        music_cog = self.bot.get_cog("MusicCog")
        if music_cog and music_cog.is_playing:
            await interaction.response.send_message(
                "Stop music first with /stop.", ephemeral=True
            )
            return
        if self.is_listening:
            self.is_listening = False
            if self._listen_task:
                self._listen_task.cancel()
                self._listen_task = None
            self._pipeline.clear_context()
            await interaction.response.send_message("Stopped listening.", ephemeral=True)
        else:
            if not self.bot.services.is_running:
                await interaction.response.send_message(
                    "Starting voice services... this may take a minute.", ephemeral=True
                )
                try:
                    await self.bot.services.ensure_running()
                except TimeoutError as e:
                    await interaction.followup.send(f"Failed to start services: {e}")
                    return
                await interaction.followup.send(
                    "Listening! Talk and I'll respond. I can also play music if you ask."
                )
            else:
                await interaction.response.send_message(
                    "Listening! Talk and I'll respond.", ephemeral=True
                )
            self.is_listening = True
            self._listen_task = asyncio.create_task(self._listen_loop(vc))

    @app_commands.command(name="recap", description="Get a summary of the conversation so far")
    async def recap(self, interaction: discord.Interaction):
        if not self._pipeline.context:
            await interaction.response.send_message(
                "No conversation to summarize yet.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        summary = await self._pipeline.summarize_context()
        if summary:
            turns = len(self._pipeline.context) // 2
            await interaction.followup.send(
                f"**Conversation recap** ({turns} turns):\n{summary}"
            )
        else:
            await interaction.followup.send("Couldn't generate a summary.")

    @app_commands.command(name="personality", description="Change bot personality")
    @app_commands.describe(preset="Personality preset name")
    @app_commands.choices(preset=[
        app_commands.Choice(name=k, value=k) for k in PERSONALITY_PRESETS
    ])
    async def personality(self, interaction: discord.Interaction, preset: str):
        if preset not in PERSONALITY_PRESETS:
            names = ", ".join(PERSONALITY_PRESETS.keys())
            await interaction.response.send_message(
                f"Unknown preset. Available: {names}", ephemeral=True
            )
            return
        self._current_personality = preset
        self._llm.system_prompt = PERSONALITY_PRESETS[preset]
        await interaction.response.send_message(
            f"Personality switched to **{preset}**.", ephemeral=True
        )

    # ── Voice-commanded music tool execution ────────────────────────

    async def _execute_tool(self, tool_call: dict):
        """Execute an LLM tool call. Queues music actions for after TTS finishes."""
        func = tool_call.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}

        music_cog = self.bot.get_cog("MusicCog")
        if not music_cog:
            log.warning("MusicCog not found, can't execute tool %s", name)
            return

        log.info("Tool call: %s(%s)", name, args)

        if name == "play_music":
            query = args.get("query", "")
            if query:
                self._pending_actions.append(
                    lambda q=query: music_cog.voice_play(self._current_guild, q)
                )
        elif name == "skip_song":
            self._pending_actions.append(
                lambda: music_cog.voice_skip(self._current_guild)
            )
        elif name == "stop_music":
            self._pending_actions.append(
                lambda: music_cog.voice_stop(self._current_guild)
            )
        elif name == "set_volume":
            level = args.get("level", 0.5)
            music_cog.set_volume(float(level))

    # ── Listen loop with barge-in support ───────────────────────────

    async def _listen_loop(self, voice_client: discord.VoiceClient):
        utterance_buffer = bytearray()
        silence_frames = 0
        silence_limit = int(self._silence_threshold_ms / VAD_FRAME_MS)
        is_speaking = False

        try:
            while self.is_listening and voice_client.is_connected():
                if not hasattr(voice_client, "_vibebot_sink"):
                    sink = AudioSink()
                    voice_client._vibebot_sink = sink
                    voice_client.listen(sink)

                sink = voice_client._vibebot_sink

                while len(sink.buffer) >= VAD_FRAME_BYTES:
                    frame = bytes(sink.buffer[:VAD_FRAME_BYTES])
                    del sink.buffer[:VAD_FRAME_BYTES]

                    try:
                        speech = self._vad.is_speech(frame, 16000)
                    except Exception:
                        speech = False

                    if speech:
                        # Barge-in: user started talking while bot is responding
                        if self._is_responding:
                            log.info("Barge-in detected, stopping bot response")
                            self._is_responding = False
                            if voice_client.is_playing():
                                voice_client.stop()

                        is_speaking = True
                        silence_frames = 0
                        utterance_buffer.extend(frame)
                        if len(utterance_buffer) >= self._max_utterance_bytes:
                            self._dispatch_utterance(
                                bytes(utterance_buffer), voice_client
                            )
                            utterance_buffer.clear()
                            is_speaking = False
                    elif is_speaking:
                        silence_frames += 1
                        if silence_frames >= silence_limit:
                            self._dispatch_utterance(
                                bytes(utterance_buffer), voice_client
                            )
                            utterance_buffer.clear()
                            is_speaking = False
                            silence_frames = 0

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            log.info("Listen loop cancelled")
        except Exception as e:
            log.error("Listen loop error: %s", e, exc_info=True)
        finally:
            if hasattr(voice_client, "_vibebot_sink"):
                voice_client.stop_listening()
                del voice_client._vibebot_sink

    def _dispatch_utterance(self, pcm: bytes, voice_client: discord.VoiceClient):
        """Fire-and-forget utterance processing (enables barge-in)."""
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = asyncio.create_task(
            self._handle_utterance(pcm, voice_client)
        )

    async def _handle_utterance(self, pcm_16k_mono: bytes,
                                voice_client: discord.VoiceClient):
        self._current_guild = voice_client.guild
        self._pending_actions = []

        source = PCMStreamSource()
        if voice_client.is_playing():
            voice_client.stop()

        # Streaming TTS: start playback immediately, feed chunks as they arrive
        voice_client.play(source)
        self._is_responding = True

        try:
            async for audio_chunk in self._pipeline.process_utterance(pcm_16k_mono):
                if not self._is_responding:
                    break  # Barge-in happened
                source.feed(audio_chunk)
        except asyncio.CancelledError:
            pass
        finally:
            source.finish()
            self._is_responding = False

        # Wait for TTS playback to drain, then execute pending music actions
        while voice_client.is_playing():
            await asyncio.sleep(0.05)

        for action in self._pending_actions:
            try:
                await action()
            except Exception as e:
                log.warning("Pending action failed: %s", e)
        self._pending_actions = []

    async def cog_unload(self):
        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        await self._asr.close()
        await self._llm.close()


class AudioSink:
    """Captures raw PCM audio from Discord voice, converting to 16kHz mono."""

    def __init__(self):
        self.buffer = bytearray()

    def write(self, user, data):
        pcm_48k_stereo = data.pcm if hasattr(data, 'pcm') else data
        pcm_48k_mono = stereo_to_mono(pcm_48k_stereo)
        pcm_16k_mono = resample_pcm(pcm_48k_mono, from_rate=48000, to_rate=16000)
        self.buffer.extend(pcm_16k_mono)

    def cleanup(self):
        self.buffer.clear()
