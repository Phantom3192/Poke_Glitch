"""
pokemon_bot.py - Discord bot using trained AI model
"""

import os
import io
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import discord
from discord.ext import commands
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import aiohttp

# Import your database layer
from train_model import Database, PokemonFeatureExtractor, PokemonClassifier

load_dotenv()

# ============ CONFIGURATION ============
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
POKETWO_BOT_ID = int(os.getenv("POKETWO_BOT_ID", "716390085896962058"))
MODEL_PATH = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "75.0"))
AMBIGUITY_MARGIN = float(os.getenv("AMBIGUITY_MARGIN", "10.0"))
MAX_SPECIES = int(os.getenv("MAX_SPECIES", "100"))

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("pokemon_bot")

# ============ DISCORD SETUP ============
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(
    command_prefix="p!",
    intents=intents,
    help_command=None,
    allowed_mentions=discord.AllowedMentions.none(),
)

# ============ AI MATCHER ============

class PokemonMatcher:
    """
    Real-time Pokémon identifier using trained model.
    Loads model once and keeps it in memory.
    """
    
    def __init__(self, model_path: str, threshold: float = 75.0, ambiguity: float = 10.0):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.ambiguity_margin = ambiguity
        
        # Load database
        self.db = Database()
        
        # Load model
        self.model = self._load_model()
        
        # Load reference features from DB
        self._load_reference_features()
        
        log.info(f"✅ Matcher initialized: {len(self.species_list)} species, {self.total_variants} variants")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Create model with correct number of species
            from train_model import PokemonClassifier, DEVICE
            
            # Get species count from database
            stats = self.db.get_stats()
            
            # Load model
            model = PokemonClassifier(num_species=stats['total_species'])
            
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
                model.feature_extractor.load_state_dict(state_dict, strict=False)
                log.info(f"✅ Loaded model from {self.model_path}")
            else:
                log.warning(f"⚠️ Model file not found: {self.model_path}")
                log.warning("   Using untrained model (will be inaccurate!)")
            
            model.eval()
            return model
            
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_reference_features(self):
        """Load all reference features from database."""
        try:
            # Get all species and their features
            cursor = self.db._conn.cursor()
            cursor.execute("""
                SELECT species, feature_vector FROM pokemon_features
                ORDER BY species, id
            """)
            
            self.species_list = []
            self.feature_matrix = []
            
            for row in cursor.fetchall():
                species = row[0]
                feature = np.array(json.loads(row[1]))
                self.species_list.append(species)
                self.feature_matrix.append(feature)
            
            if self.feature_matrix:
                self.feature_matrix = np.vstack(self.feature_matrix)
            else:
                self.feature_matrix = None
            
            self.total_variants = len(self.species_list)
            
        except Exception as e:
            log.error(f"❌ Failed to load reference features: {e}")
            self.feature_matrix = None
            self.species_list = []
    
    def _extract_features(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract features from an image using the trained model."""
        try:
            # Load and preprocess image
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Use the same transform as training
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(img).unsqueeze(0)
                
                # Extract features
                with torch.no_grad():
                    features = self.model.feature_extractor.backbone(img_tensor)
                    features = self.model.feature_extractor.projection(features)
                    features = F.normalize(features, p=2, dim=1)
                
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            log.error(f"❌ Feature extraction failed: {e}")
            return None
    
    def _compute_similarity(self, query_features: np.ndarray) -> List[Tuple[str, float, int]]:
        """Compute cosine similarity against all reference features."""
        if self.feature_matrix is None or len(self.feature_matrix) == 0:
            return []
        
        # Normalize query
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(self.feature_matrix, query_norm)
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1]
        top_similarities = similarities[top_indices]
        
        # Aggregate by species (take max similarity per species)
        species_best = {}
        for idx, sim in zip(top_indices, top_similarities):
            species = self.species_list[idx]
            if species not in species_best or sim > species_best[species][0]:
                species_best[species] = (sim, idx)
        
        # Sort by similarity
        results = sorted(
            [(species, sim, idx) for species, (sim, idx) in species_best.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return results
    
    def identify(self, image_bytes: bytes) -> Optional[Dict]:
        """Identify a Pokémon from image bytes."""
        start_time = time.time()
        
        # Extract features
        features = self._extract_features(image_bytes)
        if features is None:
            return None
        
        # Compute similarities
        results = self._compute_similarity(features)
        if not results:
            return None
        
        # Get best match
        best_species, best_sim, best_idx = results[0]
        confidence = best_sim * 100
        
        # Check if confident
        confident = confidence >= self.threshold
        
        # Check ambiguity
        second_species = None
        second_confidence = 0
        if len(results) > 1:
            second_species, second_sim, _ = results[1]
            second_confidence = second_sim * 100
            if second_confidence > confidence - self.ambiguity_margin:
                confident = False
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "species": best_species,
            "confidence": confidence,
            "confident": confident,
            "runner_up": second_species,
            "runner_up_confidence": second_confidence,
            "inference_time": inference_time,
            "variant_count": self.total_variants
        }


# ============ DISCORD COMMANDS ============

@bot.event
async def on_ready():
    log.info("=" * 50)
    log.info("🤖 Pokémon AI Namer Bot")
    log.info("=" * 50)
    log.info(f"Logged in as: {bot.user}")
    log.info(f"Loaded {matcher.total_variants} reference images")
    log.info(f"Threshold: {matcher.threshold}%")
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
    if result["runner_up"]:
        reply += f"\n⚠️ Ambiguous: {result['runner_up']} at {result['runner_up_confidence']:.1f}%"
    
    # Send reply
    try:
        await message.reply(reply, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as e:
        log.warning(f"Failed to reply: {e}")

@bot.command(name="predict")
async def predict_match(ctx: commands.Context):
    """Predict which Pokémon is in an attached image."""
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
    
    embed.add_field(name="Threshold", value=f"{matcher.threshold}%", inline=True)
    embed.add_field(name="Inference Time", value=f"{result['inference_time']:.0f}ms", inline=True)
    embed.add_field(name="Reference Images", value=str(result["variant_count"]), inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="stats")
async def show_stats(ctx: commands.Context):
    """Show bot statistics."""
    embed = discord.Embed(
        title="📊 Pokémon AI Namer Stats",
        color=discord.Color.blue()
    )
    
    stats = matcher.db.get_stats()
    
    embed.add_field(name="Species in Database", value=str(stats['total_species']), inline=True)
    embed.add_field(name="Reference Images", value=str(stats['total_features']), inline=True)
    embed.add_field(name="Match Threshold", value=f"{matcher.threshold}%", inline=True)
    embed.add_field(name="Loaded Species", value=str(len(matcher.species_list)), inline=True)
    embed.add_field(name="Loaded Variants", value=str(matcher.total_variants), inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="threshold")
@commands.is_owner()
async def set_threshold(ctx: commands.Context, value: float = None):
    """Set the match threshold."""
    if value is None:
        await ctx.send(f"Current threshold: **{matcher.threshold}%**")
        return
    
    if value < 0 or value > 100:
        await ctx.send("❌ Threshold must be between 0 and 100")
        return
    
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
    
    embed.set_footer(text=f"Threshold: {matcher.threshold}%")
    
    await ctx.send(embed=embed)


# ============ HELPER FUNCTIONS ============

async def download_image(url: str) -> Optional[bytes]:
    """Download image from URL."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.read()
                return None
        except Exception as e:
            log.warning(f"Image download failed: {e}")
            return None

def extract_spawn_image(message: discord.Message) -> Optional[str]:
    """Extract image URL from spawn message."""
    for embed in message.embeds:
        # Check for wild spawn embed
        title = (embed.title or "").lower()
        footer = (embed.footer.text or "").lower() if embed.footer else ""
        description = (embed.description or "").lower()
        combined = f"{title} {description} {footer}"
        
        if "wild" in combined and "appeared" in combined:
            if embed.image and embed.image.url:
                return embed.image.url
            if embed.thumbnail and embed.thumbnail.url:
                return embed.thumbnail.url
    
    # Check attachments
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            return att.url
    
    return None

# ============ START BOT ============

if __name__ == "__main__":
    # Initialize matcher
    matcher = PokemonMatcher(
        model_path=MODEL_PATH,
        threshold=MATCH_THRESHOLD,
        ambiguity=AMBIGUITY_MARGIN
    )
    
    # Run bot
    bot.run(DISCORD_TOKEN)
