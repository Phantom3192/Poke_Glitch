"""
model_utils.py - Inference-only model loading
NO TRAINING CODE - just loads trained model and runs inference
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ============ CONFIGURATION ============
TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
DEVICE = torch.device("cpu")


# ============ DATABASE LAYER ============

class Database:
    """Database handler - only reads features, no writes needed."""
    
    def __init__(self):
        self.use_turso = False
        self._conn = None
        
        if TURSO_URL:
            try:
                import libsql_experimental as libsql
                self._conn = libsql.connect(TURSO_URL, auth_token=TURSO_AUTH_TOKEN)
                self.use_turso = True
                print(f"✅ Connected to Turso database")
            except Exception as e:
                print(f"⚠️ Turso connection failed: {e}, using SQLite")
        
        if not self.use_turso:
            self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            print(f"✅ Using SQLite database: {DB_PATH}")
    
    def get_all_features(self) -> Dict[str, List[np.ndarray]]:
        """Get all features grouped by species."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT species, feature_vector FROM pokemon_features ORDER BY species, id")
        
        result = {}
        for row in cursor.fetchall():
            species = row[0]
            feature = np.array(json.loads(row[1]))
            result.setdefault(species, []).append(feature)
        return result
    
    def get_species_list(self) -> List[str]:
        """Get all species names."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT species FROM species_info ORDER BY species")
        return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pokemon_features")
        total_features = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM species_info")
        total_species = cursor.fetchone()[0]
        return {"total_features": total_features, "total_species": total_species}
    
    def close(self):
        if self._conn:
            self._conn.close()


# ============ AI MODEL (Inference Only) ============

class PokemonFeatureExtractor(nn.Module):
    """
    Feature extractor - loads trained weights, runs inference only.
    NO TRAINING CODE HERE!
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # Using the same architecture as training
        self.backbone = models.shufflenet_v2_x0_5(weights=None)
        self.backbone.fc = nn.Identity()
        backbone_dim = 1024
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.to(DEVICE)
        self.eval()
    
    def load_weights(self, model_path: str):
        """Load trained weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        # Load only the feature extractor weights
        self.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded model weights from {model_path}")
    
    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        """Extract feature vector from image."""
        if img is None:
            return np.zeros(256)
        
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
            img_tensor = self.normalize(img_tensor)
            features = self.backbone(img_tensor)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            return projected.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return np.zeros(256)


# ============ MATCHER ============

class PokemonMatcher:
    """
    Pokémon identifier using trained model + database features.
    """
    
    def __init__(self, model_path: str, threshold: float = 75.0, ambiguity: float = 10.0):
        self.threshold = threshold
        self.ambiguity_margin = ambiguity
        
        # 1. Load database
        print("📂 Loading database...")
        self.db = Database()
        stats = self.db.get_stats()
        print(f"   ✅ {stats['total_species']} species, {stats['total_features']} features")
        
        # 2. Load model
        print("🧠 Loading model...")
        self.model = PokemonFeatureExtractor()
        self.model.load_weights(model_path)
        print(f"   ✅ Model loaded")
        
        # 3. Load reference features
        print("📊 Loading reference features...")
        self._load_reference_features()
        print(f"   ✅ {self.total_variants} reference variants loaded")
    
    def _load_reference_features(self):
        """Load all reference features from database."""
        features_by_species = self.db.get_all_features()
        
        self.species_list = []
        self.feature_matrix = []
        
        for species, features in features_by_species.items():
            for feature in features:
                self.species_list.append(species)
                self.feature_matrix.append(feature)
        
        if self.feature_matrix:
            self.feature_matrix = np.vstack(self.feature_matrix)
        else:
            self.feature_matrix = None
        
        self.total_variants = len(self.species_list)
        self.species_set = set(self.species_list)
    
    def _compute_similarity(self, query_features: np.ndarray) -> List[Tuple[str, float]]:
        """Compute cosine similarity against all reference features."""
        if self.feature_matrix is None or len(self.feature_matrix) == 0:
            return []
        
        # Normalize query
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        # Compute all similarities
        similarities = np.dot(self.feature_matrix, query_norm)
        
        # Get top indices
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
            [(species, sim) for species, (sim, idx) in species_best.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return results
    
    def identify(self, image_bytes: bytes) -> Optional[Dict]:
        """Identify a Pokémon from image bytes."""
        import time
        start_time = time.time()
        
        # 1. Load image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"⚠️ Failed to load image: {e}")
            return None
        
        # 2. Extract features
        features = self.model.extract(img)
        if np.all(features == 0):
            return None
        
        # 3. Compute similarities
        results = self._compute_similarity(features)
        if not results:
            return None
        
        # 4. Get best match
        best_species, best_sim = results[0]
        confidence = best_sim * 100
        
        # 5. Check confidence
        confident = confidence >= self.threshold
        
        # 6. Check ambiguity
        runner_up = None
        runner_up_conf = 0
        if len(results) > 1:
            runner_up, runner_up_sim = results[1]
            runner_up_conf = runner_up_sim * 100
            if runner_up_conf > confidence - self.ambiguity_margin:
                confident = False
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "species": best_species,
            "confidence": confidence,
            "confident": confident,
            "runner_up": runner_up,
            "runner_up_confidence": runner_up_conf,
            "inference_time": inference_time,
            "total_variants": self.total_variants
        }
