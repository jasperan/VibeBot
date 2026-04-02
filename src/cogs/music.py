import asyncio
import logging
from collections import deque

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("vibebot.music")

FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}


class MusicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        mc = bot.config["music"]
        self._max_queue = mc["max_queue_size"]
        self._volume = mc["default_volume"]
        self.queue: deque[dict] = deque(maxlen=self._max_queue)
        self.is_playing = False
        self.now_playing: dict | None = None

    def _clear_state(self):
        self.queue.clear()
        self.is_playing = False
        self.now_playing = None

    def set_volume(self, level: float):
        """Set volume level (0.0 to 1.0). Applies immediately if playing."""
        self._volume = max(0.0, min(1.0, level))

    async def voice_play(self, guild: discord.Guild, query: str):
        """Play a track triggered by voice command (no interaction needed)."""
        vc = guild.voice_client
        if not vc:
            return
        track = await self._search_youtube(query)
        if not track:
            log.info("Voice play: couldn't find track for '%s'", query)
            return
        track["requester"] = "VibeBot (voice)"
        if vc.is_playing():
            self.queue.append(track)
            log.info("Voice play: queued '%s'", track["title"])
        else:
            self.is_playing = True
            self.now_playing = track
            source = discord.FFmpegPCMAudio(track["url"], **FFMPEG_OPTIONS)
            source = discord.PCMVolumeTransformer(source, volume=self._volume)
            vc.play(source, after=lambda e: self._play_next(guild, e))
            log.info("Voice play: now playing '%s'", track["title"])

    async def voice_skip(self, guild: discord.Guild):
        """Skip triggered by voice command."""
        vc = guild.voice_client
        if vc and vc.is_playing():
            vc.stop()

    async def voice_stop(self, guild: discord.Guild):
        """Stop triggered by voice command."""
        vc = guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
        self._clear_state()

    def _check_listening(self) -> bool:
        voice_cog = self.bot.get_cog("VoiceCog")
        return voice_cog is not None and voice_cog.is_listening

    async def _search_youtube(self, query: str) -> dict | None:
        loop = asyncio.get_running_loop()
        try:
            import yt_dlp
            ydl_opts = {
                "format": "bestaudio/best",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "default_search": "ytsearch",
                "extract_flat": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(
                    None, lambda: ydl.extract_info(query, download=False)
                )
                if "entries" in info:
                    info = info["entries"][0]
                return {
                    "title": info.get("title", "Unknown"),
                    "url": info["url"],
                    "webpage_url": info.get("webpage_url", ""),
                    "duration": info.get("duration", 0),
                }
        except Exception as e:
            log.warning("yt-dlp search failed: %s", e)
            return None

    def _play_next(self, guild: discord.Guild, error=None):
        if error:
            log.warning("Player error: %s", error)
        if self.queue:
            track = self.queue.popleft()
            self.now_playing = track
            vc = guild.voice_client
            if vc:
                source = discord.FFmpegPCMAudio(track["url"], **FFMPEG_OPTIONS)
                source = discord.PCMVolumeTransformer(source, volume=self._volume)
                vc.play(source, after=lambda e: self._play_next(guild, e))
        else:
            self.is_playing = False
            self.now_playing = None

    @app_commands.command(name="play", description="Play a song from YouTube")
    @app_commands.describe(query="Song name or YouTube URL")
    async def play(self, interaction: discord.Interaction, query: str):
        if self._check_listening():
            await interaction.response.send_message(
                "Disable listening first with /listen.", ephemeral=True
            )
            return
        vc = interaction.guild.voice_client
        if not vc:
            if interaction.user.voice:
                vc = await interaction.user.voice.channel.connect()
            else:
                await interaction.response.send_message(
                    "You need to be in a voice channel.", ephemeral=True
                )
                return
        await interaction.response.defer()
        track = await self._search_youtube(query)
        if not track:
            await interaction.followup.send("Couldn't find that track.")
            return
        track["requester"] = interaction.user.display_name
        if vc.is_playing():
            self.queue.append(track)
            await interaction.followup.send(
                f"Queued: **{track['title']}** (#{len(self.queue)} in queue)"
            )
        else:
            self.is_playing = True
            self.now_playing = track
            source = discord.FFmpegPCMAudio(track["url"], **FFMPEG_OPTIONS)
            source = discord.PCMVolumeTransformer(source, volume=self._volume)
            vc.play(source, after=lambda e: self._play_next(interaction.guild, e))
            await interaction.followup.send(f"Now playing: **{track['title']}**")

    @app_commands.command(name="skip", description="Skip the current song")
    async def skip(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc or not vc.is_playing():
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        vc.stop()
        await interaction.response.send_message("Skipped.", ephemeral=True)

    @app_commands.command(name="queue", description="Show the music queue")
    async def show_queue(self, interaction: discord.Interaction):
        if not self.queue and not self.now_playing:
            await interaction.response.send_message("Queue is empty.", ephemeral=True)
            return
        lines = []
        if self.now_playing:
            lines.append(f"**Now playing:** {self.now_playing['title']}")
        for i, track in enumerate(self.queue, 1):
            lines.append(f"{i}. {track['title']}")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="pause", description="Pause the current song")
    async def pause(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and vc.is_playing():
            vc.pause()
            await interaction.response.send_message("Paused.", ephemeral=True)
        else:
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)

    @app_commands.command(name="resume", description="Resume the paused song")
    async def resume(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and vc.is_paused():
            vc.resume()
            await interaction.response.send_message("Resumed.", ephemeral=True)
        else:
            await interaction.response.send_message("Nothing is paused.", ephemeral=True)

    @app_commands.command(name="stop", description="Stop music and clear queue")
    async def stop(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
        self._clear_state()
        await interaction.response.send_message("Stopped and cleared queue.", ephemeral=True)

    @app_commands.command(name="np", description="Show now playing")
    async def now_playing_cmd(self, interaction: discord.Interaction):
        if not self.now_playing:
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        track = self.now_playing
        duration = track.get("duration", 0)
        mins, secs = divmod(duration, 60)
        await interaction.response.send_message(
            f"**{track['title']}** [{mins}:{secs:02d}] "
            f"(requested by {track.get('requester', 'unknown')})",
            ephemeral=True,
        )

    @app_commands.command(name="volume", description="Set music volume (0-100)")
    @app_commands.describe(level="Volume level 0-100")
    async def volume(self, interaction: discord.Interaction, level: int):
        if level < 0 or level > 100:
            await interaction.response.send_message(
                "Volume must be between 0 and 100.", ephemeral=True
            )
            return
        self._volume = level / 100.0
        vc = interaction.guild.voice_client
        if vc and vc.source and isinstance(vc.source, discord.PCMVolumeTransformer):
            vc.source.volume = self._volume
        await interaction.response.send_message(
            f"Volume set to {level}%", ephemeral=True
        )
