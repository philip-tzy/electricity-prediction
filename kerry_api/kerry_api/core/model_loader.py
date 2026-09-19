"""
Model Loader — Singleton
========================
Memuat model.pkl, preprocessor.pkl, dan bom_routing.json
sekali saat server startup, lalu menyimpannya di memori.

Semua router mengakses model melalui ModelLoader.get() —
tidak pernah load ulang dari disk per request.
"""

import os
import json
import joblib
from pathlib import Path

# Path ke file model — sesuaikan jika berbeda
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "model_files" / "model.pkl"
PREP_PATH   = BASE_DIR / "model_files" / "preprocessor.pkl"
BOM_PATH    = BASE_DIR / "model_files" / "bom_routing.json"


class ModelLoader:
    """
    Singleton holder untuk semua artefak model.
    Gunakan ModelLoader.get() untuk mengakses dari router.
    """
    _model        = None
    _preprocessor = None
    _bom_routing  = None
    _loaded       = False

    @classmethod
    def load(cls):
        """Dipanggil sekali saat app startup."""
        if cls._loaded:
            return

        # Validasi file tersedia
        for path, label in [
            (MODEL_PATH, "model.pkl"),
            (PREP_PATH,  "preprocessor.pkl"),
            (BOM_PATH,   "bom_routing.json"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"File tidak ditemukan: {path}\n"
                    f"Pastikan {label} sudah diletakkan di folder model_files/"
                )

        cls._model        = joblib.load(MODEL_PATH)
        cls._preprocessor = joblib.load(PREP_PATH)

        with open(BOM_PATH, encoding="utf-8") as f:
            cls._bom_routing = json.load(f)

        cls._loaded = True
        print(f"  [OK] model.pkl        dimuat ({MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  [OK] preprocessor.pkl dimuat")
        print(f"  [OK] bom_routing.json dimuat "
              f"({len(cls._bom_routing['products'])} produk, "
              f"{len(cls._bom_routing['machines'])} mesin)")

    @classmethod
    def get(cls):
        """Kembalikan tuple (model, preprocessor, bom_routing)."""
        if not cls._loaded:
            cls.load()
        return cls._model, cls._preprocessor, cls._bom_routing
