import asyncio
import logging
import sys

import discord
from discord.ext import commands
import yaml

log = logging.getLogger("vibebot")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class VibeBotClient(commands.Bot):
    def __init__(self, config: dict):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.mode_lock = asyncio.Lock()

    async def setup_hook(self):
        await self._load_cogs()
        await self.tree.sync()
        log.info("Slash commands synced")

    async def _load_cogs(self):
        from src.cogs.voice import VoiceCog
        from src.cogs.music import MusicCog
        await self.add_cog(VoiceCog(self))
        await self.add_cog(MusicCog(self))
        log.info("Cogs loaded: VoiceCog, MusicCog")

    async def on_ready(self):
        log.info(f"VibeBot online as {self.user} (id={self.user.id})")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = load_config()
    bot = VibeBotClient(config)
    bot.run(config["discord"]["token"], log_handler=None)


if __name__ == "__main__":
    main()
