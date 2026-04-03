import asyncio
import logging
import sys

import discord
from discord.ext import commands
import yaml

log = logging.getLogger("vibebot")
PLACEHOLDER_DISCORD_TOKEN = "YOUR_DISCORD_BOT_TOKEN"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_runtime_config(config: dict) -> None:
    token = config.get("discord", {}).get("token", "").strip()
    if not token or token == PLACEHOLDER_DISCORD_TOKEN:
        raise ValueError(
            "config.yaml has no real Discord bot token. Update discord.token before running VibeBot."
        )


class VibeBotClient(commands.Bot):
    def __init__(self, config: dict):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.mode_lock = asyncio.Lock()

        from src.service_manager import ServiceManager
        self.services = ServiceManager(config)

    async def setup_hook(self):
        await self._load_cogs()
        await self.tree.sync()
        log.info("Slash commands synced")

    async def _load_cogs(self):
        from src.cogs.voice import VoiceCog
        from src.cogs.music import MusicCog
        from src.cogs.admin import AdminCog
        await self.add_cog(VoiceCog(self))
        await self.add_cog(MusicCog(self))
        await self.add_cog(AdminCog(self))
        log.info("Cogs loaded: VoiceCog, MusicCog, AdminCog")

    async def on_ready(self):
        log.info(f"VibeBot online as {self.user} (id={self.user.id})")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = load_config()
    validate_runtime_config(config)
    bot = VibeBotClient(config)
    bot.run(config["discord"]["token"], log_handler=None)


if __name__ == "__main__":
    main()
