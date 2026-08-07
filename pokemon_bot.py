"""
pokemon_bot.py - Discord bot using trained AI model
"""

import os
import io
import logging
import asyncio
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv
import aiohttp
from PIL import Image

# Import ONLY inference code - no training!
from model_utils import PokemonMatcher

load_dotenv()

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("pokemon_bot")

# ============ CONFIGURATION ============
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise SystemExit("❌ DISCORD_TOKEN not set!")

POKETWO_BOT_ID = int(os.getenv("POKETWO_BOT_ID", "716390085896962058"))
MODEL_PATH = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "75.0"))
AMBIGUITY_MARGIN = float(os.getenv("AMBIGUITY_MARGIN", "10.0"))

# ============ DISCORD SETUP ============
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="p!",
    intents=intents,
    help_command=None,
    allowed_mentions=discord.AllowedMentions.none(),
)

# ============ GLOBALS ============
session: Optional[aiohttp.ClientSession] = None
matcher: Optional[PokemonMatcher] = None


# ============ HELPER FUNCTIONS ============

async def get_session() -> aiohttp.ClientSession:
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()
    return session


async def download_image(url: str) -> Optional[bytes]:
    """Download image from URL."""
    sess = await get_session()
    try:
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.read()
            return None
    except Exception as e:
        log.warning(f"Image download failed: {e}")
        return None


def extract_spawn_image(message: discord.Message) -> Optional[str]:
    """Extract image URL from spawn message."""
    for embed in message.embeds:
        title = (embed.title or "").lower()
        footer = (embed.footer.text or "").lower() if embed.footer else ""
        description = (embed.description or "").lower()
        combined = f"{title} {description} {footer}"
        
        if "wild" in combined and "appeared" in combined:
            if embed.image and embed.image.url:
                return embed.image.url
            if embed.thumbnail and embed.thumbnail.url:
                return embed.thumbnail.url
    
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            return att.url
    
    return None


# ============ BOT EVENTS ============

@bot.event
async def on_ready():
    global matcher
    
    log.info("=" * 50)
    log.info("🤖 Pokémon AI Namer Bot")
    log.info("=" * 50)
    log.info(f"Logged in as: {bot.user}")
    
    # Initialize matcher
    log.info("🔄 Initializing AI matcher...")
    try:
        matcher = PokemonMatcher(
            model_path=MODEL_PATH,
            threshold=MATCH_THRESHOLD,
            ambiguity=AMBIGUITY_MARGIN
        )
        log.info("✅ AI matcher ready!")
    except Exception as e:
        log.error(f"❌ Failed to initialize matcher: {e}")
        return
    
    log.info("=" * 50)
    log.info("✅ Bot is ready!")


@bot.event
async def on_message(message: discord.Message):
    """Process incoming messages."""
    await bot.process_commands(message)
    
    # Ignore own messages
    if message.author.id == bot.user.id:
        return
    
    # Check if it's a Poketwo spawn
    if message.author.id == POKETWO_BOT_ID:
        image_url = extract_spawn_image(message)
        if image_url:
            await process_spawn(message, image_url)


async def process_spawn(message: discord.Message, image_url: str):
    """Process a spawn message with AI identification."""
    if matcher is None:
        return
    
    # Download image
    image_bytes = await download_image(image_url)
    if not image_bytes:
        return
    
    # Identify
    try:
        result = await asyncio.to_thread(matcher.identify, image_bytes)
    except Exception as e:
        log.error(f"Identification failed: {e}")
        return
    
    if result is None or not result["confident"]:
        return
    
    # Build reply
    reply = f"**{result['species']}** — {result['confidence']:.1f}%"
    
    # Send reply
    try:
        await message.reply(reply, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as e:
        log.warning(f"Failed to reply: {e}")


# ============ COMMANDS ============

@bot.command(name="predict")
async def predict_match(ctx: commands.Context):
    """Predict which Pokémon is in an attached image."""
    if matcher is None:
        await ctx.send("❌ AI matcher not ready yet!")
        return
    
    image_bytes = None
    
    # Check attachments
    for att in ctx.message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            try:
                image_bytes = await att.read()
                break
            except Exception:
                continue
    
    # Check replied message
    if image_bytes is None and ctx.message.reference:
        try:
            replied = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            for att in replied.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    try:
                        image_bytes = await att.read()
                        break
                    except Exception:
                        continue
            if image_bytes is None:
                image_url = extract_spawn_image(replied)
                if image_url:
                    image_bytes = await download_image(image_url)
        except Exception:
            pass
    
    if image_bytes is None:
        await ctx.send("⚠️ Please attach an image or reply to a message with an image.")
        return
    
    await ctx.send("🔍 Processing image...")
    
    try:
        result = await asyncio.to_thread(matcher.identify, image_bytes)
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")
        return
    
    if result is None:
        await ctx.send("❌ No Pokémon recognized.")
        return
    
    embed = discord.Embed(
        title="🔍 Prediction Result",
        color=discord.Color.green() if result["confident"] else discord.Color.orange()
    )
    
    embed.add_field(name="Best Guess", value=f"**{result['species']}**", inline=True)
    embed.add_field(name="Confidence", value=f"{result['confidence']:.1f}%", inline=True)
    embed.add_field(
        name="Status",
        value="✅ Confident" if result["confident"] else "⚠️ Uncertain",
        inline=True
    )
    
    if result["runner_up"]:
        embed.add_field(
            name="Runner-up",
            value=f"{result['runner_up']} ({result['runner_up_confidence']:.1f}%)",
            inline=False
        )
    
    embed.add_field(name="Threshold", value=f"{MATCH_THRESHOLD}%", inline=True)
    embed.add_field(name="Inference Time", value=f"{result['inference_time']:.0f}ms", inline=True)
    embed.add_field(name="Reference Images", value=str(result["total_variants"]), inline=True)
    
    await ctx.send(embed=embed)


@bot.command(name="stats")
async def show_stats(ctx: commands.Context):
    """Show bot statistics."""
    if matcher is None:
        await ctx.send("❌ AI matcher not ready yet!")
        return
    
    stats = matcher.db.get_stats()
    
    embed = discord.Embed(
        title="📊 Pokémon AI Namer Stats",
        color=discord.Color.blue()
    )
    
    embed.add_field(name="Species in Database", value=str(stats['total_species']), inline=True)
    embed.add_field(name="Reference Images", value=str(stats['total_features']), inline=True)
    embed.add_field(name="Match Threshold", value=f"{MATCH_THRESHOLD}%", inline=True)
    embed.add_field(name="Loaded Variants", value=str(matcher.total_variants), inline=True)
    
    await ctx.send(embed=embed)


@bot.command(name="threshold")
@commands.is_owner()
async def set_threshold(ctx: commands.Context, value: float = None):
    
    """Set the match threshold."""
    
    global MATCH_THRESHOLD 
    
    if value is None:
        await ctx.send(f"Current threshold: **{MATCH_THRESHOLD}%**")
        return
    
    if value < 0 or value > 100:
        await ctx.send("❌ Threshold must be between 0 and 100")
        return
    
    global MATCH_THRESHOLD
    MATCH_THRESHOLD = value
    if matcher:
        matcher.threshold = value
    await ctx.send(f"✅ Threshold set to **{value}%**")


@bot.command(name="help")
async def show_help(ctx: commands.Context):
    """Show help message."""
    embed = discord.Embed(
        title="🤖 Pokémon AI Namer Bot",
        description="Identify Pokémon from images using AI",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="p!predict",
        value="Predict which Pokémon is in an image.\nAttach an image or reply to one.",
        inline=False
    )
    embed.add_field(
        name="p!stats",
        value="Show bot statistics.",
        inline=False
    )
    embed.add_field(
        name="p!threshold",
        value="Show or set match threshold. (Bot owner only)",
        inline=False
    )
    embed.add_field(
        name="p!help",
        value="Show this help message.",
        inline=False
    )
    
    embed.set_footer(text=f"Threshold: {MATCH_THRESHOLD}%")
    
    await ctx.send(embed=embed)


# ============ CLEANUP ============

@bot.event
async def on_close():
    global session
    if session and not session.closed:
        await session.close()
    if matcher:
        matcher.db.close()


# ============ START ============

if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN)
    finally:
        if session and not session.closed:
            asyncio.run(session.close())
