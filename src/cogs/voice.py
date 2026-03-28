import asyncio
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


class VoiceCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.is_listening = False
        self._listen_task: asyncio.Task | None = None

        vc = bot.config["voice"]
        lc = bot.config["llm"]

        self._asr = ASRClient(base_url=vc["asr_url"])
        self._llm = LLMClient(
            base_url=lc["base_url"],
            model=lc["model"],
            max_tokens=lc["max_tokens"],
            system_prompt=lc["system_prompt"],
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
        self._vad = webrtcvad.Vad(vc["vad_aggressiveness"])
        self._silence_threshold_ms = vc["silence_threshold_ms"]
        self._max_utterance_bytes = int(vc["max_utterance_seconds"] * 16000 * 2)

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
        self._pipeline.clear_context()
        music_cog = self.bot.get_cog("MusicCog")
        if music_cog:
            music_cog._clear_state()
        await vc.disconnect()
        # Shutdown voice services to free GPU
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
            # Lazy-start voice services if not running
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
                    "Listening! Talk and I'll respond."
                )
            else:
                await interaction.response.send_message(
                    "Listening! Talk and I'll respond.", ephemeral=True
                )
            self.is_listening = True
            self._listen_task = asyncio.create_task(self._listen_loop(vc))

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
                        is_speaking = True
                        silence_frames = 0
                        utterance_buffer.extend(frame)
                        if len(utterance_buffer) >= self._max_utterance_bytes:
                            await self._handle_utterance(bytes(utterance_buffer), voice_client)
                            utterance_buffer.clear()
                            is_speaking = False
                    elif is_speaking:
                        silence_frames += 1
                        if silence_frames >= silence_limit:
                            await self._handle_utterance(bytes(utterance_buffer), voice_client)
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

    async def _handle_utterance(self, pcm_16k_mono: bytes, voice_client: discord.VoiceClient):
        source = PCMStreamSource()
        async for audio_chunk in self._pipeline.process_utterance(pcm_16k_mono):
            source.feed(audio_chunk)
        source.finish()
        if voice_client.is_playing():
            voice_client.stop()
        voice_client.play(source)

    async def cog_unload(self):
        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
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
