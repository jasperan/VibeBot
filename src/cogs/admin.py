import logging
import time

import discord
from discord import app_commands
from discord.ext import commands
import httpx

log = logging.getLogger("vibebot.admin")


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="status", description="Check service health and bot status")
    async def status(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        lines = []

        # Bot info
        voice_cog = self.bot.get_cog("VoiceCog")
        music_cog = self.bot.get_cog("MusicCog")
        listening = voice_cog.is_listening if voice_cog else False
        playing = music_cog.is_playing if music_cog else False
        personality = getattr(voice_cog, "_current_personality", "default")

        lines.append(f"**VibeBot Status**")
        lines.append(f"Guilds: {len(self.bot.guilds)} | Voice connections: {len(self.bot.voice_clients)}")
        lines.append(f"Listening: {'Yes' if listening else 'No'} | Music: {'Playing' if playing else 'Idle'}")
        lines.append(f"Personality: {personality}")
        lines.append("")
        lines.append("**Services:**")

        # Check each service
        services_config = self.bot.config.get("services", {})
        for name, svc in services_config.items():
            health_url = svc.get("health_url", "")
            if not health_url:
                lines.append(f"  {name}: no health URL configured")
                continue

            start = time.monotonic()
            healthy = await self._check_health(health_url)
            latency_ms = int((time.monotonic() - start) * 1000)

            status_icon = "UP" if healthy else "DOWN"
            status_text = f"  {name}: {status_icon}"
            if healthy:
                status_text += f" ({latency_ms}ms)"

            proc = self.bot.services._processes.get(name)
            if proc and proc.poll() is None:
                status_text += f" [PID {proc.pid}]"
            elif proc:
                status_text += f" [exited: {proc.returncode}]"

            lines.append(status_text)

        await interaction.followup.send("\n".join(lines))

    async def _check_health(self, url: str) -> bool:
        if url.startswith("ws://") or url.startswith("wss://"):
            try:
                import websockets
                async with websockets.connect(url, close_timeout=2) as ws:
                    return True
            except Exception:
                return False
        else:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    return resp.status_code == 200
            except Exception:
                return False
