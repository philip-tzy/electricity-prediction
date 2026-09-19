"""
Router: Schedule
POST /api/predict-schedule              → Input order → output jadwal optimal + KPI
GET  /api/schedule/{id}                 → Ambil jadwal dari cache berdasarkan order_id
POST /api/schedule/{id}/approve         → Production Planner menyetujui jadwal (PRD §4)
DELETE /api/schedule/{id}               → Hapus jadwal dari cache
"""

import uuid
from typing import Dict
from fastapi import APIRouter, HTTPException

from core.model_loader import ModelLoader
from core.predictor import expand_bom, run_optimizer, build_schedule
from schemas.schedule import ScheduleRequest, ScheduleResponse

router = APIRouter()

# ── In-memory cache jadwal ───────────────────────────────────────────────────
# Di production, ganti dengan Redis atau database
_schedule_cache: Dict[str, dict] = {}


@router.post(
    "/predict-schedule",
    response_model=ScheduleResponse,
    summary="Generate jadwal produksi optimal",
    description="""
**Endpoint utama sistem AI.**

Alur yang terjadi setelah request diterima:
1. Validasi input order (product_id valid, quantity > 0)
2. Ekspansi BOM — setiap order diurai menjadi sesi-sesi mesin sesuai rute
3. Optimizer mengevaluasi semua kombinasi (mesin × shift) menggunakan model XGBoost
4. Jadwal optimal disusun lengkap dengan instruksi operator dan WIP notes
5. KPI dihitung: total kWh, cost per MT, hari operasional, penghematan vs manual
    """,
)
def predict_schedule(request: ScheduleRequest):
    """
    Menerima order dari Supply Chain dan mengembalikan jadwal mesin optimal.
    """
    model, preprocessor, bom_routing = ModelLoader.get()

    available_products = set(bom_routing["products"].keys())
    invalid = [
        o.product_id for o in request.orders
        if o.product_id not in available_products
    ]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "message"  : f"Product ID tidak ditemukan di BOM: {invalid}",
                "available": sorted(available_products),
            },
        )

    total_qty = sum(o.quantity_mt for o in request.orders)
    if total_qty <= 0:
        raise HTTPException(
            status_code=422,
            detail="Total quantity_mt harus lebih dari 0"
        )

    orders_dict = [
        {"product_id": o.product_id, "quantity_mt": o.quantity_mt}
        for o in request.orders
    ]
    sessions = expand_bom(orders_dict, bom_routing)

    if not sessions:
        raise HTTPException(
            status_code=500,
            detail="Gagal mengekspansi BOM — tidak ada sesi yang dihasilkan"
        )

    optimized_sessions = run_optimizer(
        sessions, bom_routing, model, preprocessor, month_str=request.month
    )
    result = build_schedule(optimized_sessions, request.month, bom_routing)

    order_id = str(uuid.uuid4())[:8].upper()
    result["order_id"] = order_id
    _schedule_cache[order_id] = result

    return result


@router.get(
    "/schedule/{order_id}",
    response_model=ScheduleResponse,
    summary="Ambil jadwal dari cache",
)
def get_schedule(order_id: str):
    result = _schedule_cache.get(order_id.upper())
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Jadwal dengan order_id '{order_id}' tidak ditemukan. "
                   f"Cache bersifat sementara — generate ulang jika server restart."
        )
    return result


@router.post(
    "/schedule/{order_id}/approve",
    summary="Setujui jadwal (Production Planner)",
    description="PRD §4: Production Planner menyetujui jadwal sebelum dieksekusi operator.",
)
def approve_schedule(order_id: str):
    key = order_id.upper()
    result = _schedule_cache.get(key)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Jadwal '{order_id}' tidak ditemukan",
        )
    result["approval_status"] = "approved"
    for row in result.get("schedule", []):
        row["status"] = "approved"
    _schedule_cache[key] = result
    return {
        "order_id"       : key,
        "approval_status": "approved",
        "message"        : "Jadwal disetujui Production Planner dan siap dieksekusi operator.",
    }


@router.delete(
    "/schedule/{order_id}",
    summary="Hapus jadwal dari cache",
)
def delete_schedule(order_id: str):
    key = order_id.upper()
    if key not in _schedule_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Jadwal '{order_id}' tidak ditemukan"
        )
    del _schedule_cache[key]
    return {"message": f"Jadwal '{order_id}' berhasil dihapus"}


@router.get(
    "/schedule",
    summary="Daftar semua jadwal dalam cache",
)
def list_schedules():
    if not _schedule_cache:
        return {"schedules": [], "total": 0}

    summaries = []
    for oid, data in _schedule_cache.items():
        summaries.append({
            "order_id"       : oid,
            "month"          : data.get("month"),
            "generated_at"   : data.get("generated_at"),
            "approval_status": data.get("approval_status", "pending"),
            "total_sessions" : data.get("total_sessions"),
            "kpi": {
                "total_kwh"         : data["kpi"]["total_kwh"],
                "kwh_per_mt"        : data["kpi"]["kwh_per_mt"],
                "operating_days"    : data["kpi"]["operating_days"],
                "saving_pct"        : data["kpi"]["saving_pct"],
                "saving_rp"         : data["kpi"].get("saving_rp", 0),
                "prd_kpi_target_met": data["kpi"]["prd_kpi_target_met"],
            },
        })

    return {
        "schedules": summaries,
        "total"    : len(summaries),
    }
