"""
================================================================================
Kerry AI Production Scheduling System — FastAPI Backend
================================================================================
Entry point utama. Jalankan dengan:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Dashboard:  http://localhost:8000/dashboard
API docs:   http://localhost:8000/docs

Endpoint yang tersedia:
    GET  /api/health                Health check
    GET  /api/products              Daftar semua produk + rute BOM
    GET  /api/machines              Daftar semua mesin
    POST /api/predict-schedule      Input order → output jadwal optimal + KPI
    GET  /api/schedule/{order_id}   Ambil jadwal berdasarkan ID (dari cache)
    GET  /api/kpi/summary           Ringkasan KPI bulan berjalan
================================================================================
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from routers import products, machines, schedule, kpi
from core.model_loader import ModelLoader

# Cari folder frontend secara robust (bekerja dari package / launcher / cwd berbeda)
def _resolve_frontend_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "frontend",                 # kerry_api/kerry_api → kerry/frontend
        here.parent.parent.parent / "frontend",          # jika nested beda
        Path.cwd() / "frontend",
        Path.cwd().parent / "frontend",
        Path.cwd().parent.parent / "frontend",
    ]
    for c in candidates:
        if (c / "kerry_dashboard.html").is_file():
            return c
    return candidates[0]

FRONTEND_DIR = _resolve_frontend_dir()
DASHBOARD_FILE = FRONTEND_DIR / "kerry_dashboard.html"

# ── Inisialisasi aplikasi ─────────────────────────────────────────────────────
app = FastAPI(
    title="Kerry AI Production Scheduling API",
    description=(
        "Backend API untuk sistem penjadwalan produksi berbasis AI. "
        "Menerima target order bulanan dari Supply Chain, memprediksi "
        "konsumsi listrik menggunakan XGBoost, dan menghasilkan jadwal "
        "mesin yang paling efisien."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI → buka http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc     → buka http://localhost:8000/redoc
)

# ── CORS — izinkan frontend mengakses API (termasuk file:// / port lain) ──────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wajib False jika allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model saat server startup ───────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    ModelLoader.load()
    print("=" * 55)
    print("  Kerry AI Production Scheduling API")
    print("  Model XGBoost berhasil dimuat ke memori")
    print("  Dashboard: http://localhost:8000/dashboard")
    print("  Docs:      http://localhost:8000/docs")
    print("=" * 55)

# ── Daftarkan semua router ───────────────────────────────────────────────────
app.include_router(products.router,  prefix="/api", tags=["Products & BOM"])
app.include_router(machines.router,  prefix="/api", tags=["Machines"])
app.include_router(schedule.router,  prefix="/api", tags=["Schedule"])
app.include_router(kpi.router,       prefix="/api", tags=["KPI"])

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
@app.get("/", tags=["Health"], include_in_schema=False)
def root():
    return {
        "status"   : "ok",
        "service"  : "Kerry AI Production Scheduling API",
        "version"  : "1.0.0",
        "docs"     : "/docs",
        "dashboard": "/dashboard",
    }

# ── Frontend dashboard ───────────────────────────────────────────────────────
@app.get("/dashboard", tags=["Frontend"], include_in_schema=False)
@app.get("/index.html", tags=["Frontend"], include_in_schema=False)
def dashboard():
    """Sajikan dashboard HTML dari folder frontend/."""
    if not DASHBOARD_FILE.is_file():
        return {
            "error": f"Dashboard tidak ditemukan di {DASHBOARD_FILE}. "
                     "Pastikan folder frontend/ ada di root project.",
            "tried": str(DASHBOARD_FILE),
        }
    return FileResponse(
        DASHBOARD_FILE,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/app", tags=["Frontend"], include_in_schema=False)
def app_redirect():
    return RedirectResponse(url="/dashboard")


# Static assets di frontend/ (jika nanti ada CSS/JS terpisah)
if FRONTEND_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend-static",
    )
