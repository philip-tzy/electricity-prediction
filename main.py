"""
Launcher — agar `uvicorn main:app` bisa dijalankan dari root project kerry/.

App sesungguhnya ada di kerry_api/kerry_api/main.py (logika tidak diubah).
"""
from pathlib import Path
import importlib.util
import sys

_APP_DIR = Path(__file__).resolve().parent / "kerry_api" / "kerry_api"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

_spec = importlib.util.spec_from_file_location(
    "kerry_api_main",
    _APP_DIR / "main.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["kerry_api_main"] = _module
_spec.loader.exec_module(_module)

app = _module.app
