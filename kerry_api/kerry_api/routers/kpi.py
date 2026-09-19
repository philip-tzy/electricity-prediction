"""
Router: KPI
GET /api/kpi/summary        → Ringkasan KPI dari semua jadwal dalam cache
GET /api/kpi/{order_id}     → KPI detail satu jadwal
"""

from fastapi import APIRouter, HTTPException
from routers.schedule import _schedule_cache

router = APIRouter()


@router.get("/kpi/summary", summary="Ringkasan KPI semua jadwal")
def get_kpi_summary():
    """
    Mengembalikan agregasi KPI dari semua jadwal yang ada di cache.

    Dipakai frontend untuk:
    - Panel KPI di halaman utama dashboard
    - Laporan penghematan ke manajemen (PRD §4 & §6.4)
    """
    if not _schedule_cache:
        return {
            "message": "Belum ada jadwal yang digenerate. "
                       "Gunakan POST /api/predict-schedule terlebih dahulu.",
            "total_schedules": 0,
        }

    all_kwh    = [d["kpi"]["total_kwh"] for d in _schedule_cache.values()]
    all_mt     = [d["kpi"]["total_mt"] for d in _schedule_cache.values()]
    all_days   = [d["kpi"]["operating_days"] for d in _schedule_cache.values()]
    all_saving = [d["kpi"]["saving_pct"] for d in _schedule_cache.values()]
    all_rp     = [d["kpi"].get("saving_rp", 0) for d in _schedule_cache.values()]
    prd_pass   = all(d["kpi"]["prd_kpi_target_met"] for d in _schedule_cache.values())

    return {
        "total_schedules": len(_schedule_cache),
        "aggregate": {
            "total_kwh"         : round(sum(all_kwh), 2),
            "total_mt"          : round(sum(all_mt), 2),
            "avg_kwh_per_mt"    : round(sum(all_kwh) / sum(all_mt), 4) if sum(all_mt) > 0 else 0,
            "avg_operating_days": round(sum(all_days) / len(all_days), 1),
            "avg_saving_pct"    : round(sum(all_saving) / len(all_saving), 1),
            "total_saving_rp"   : round(sum(all_rp), 0),
            "prd_target_met"    : prd_pass,
        },
        "prd_acceptance": {
            "criteria": [
                "Cost per MT < 140 kWh/MT (PRD: 0.14 MWh/MT)",
                "Hari operasional AI ≤ 70% baseline manual (turun ≥ 30%)",
                "100% order terpenuhi",
                "Constraint mesin/BOM tidak dilanggar",
            ],
            "status": "PASS" if prd_pass else "FAIL",
        },
    }


@router.get("/kpi/{order_id}", summary="KPI detail satu jadwal")
def get_kpi_by_order(order_id: str):
    """
    KPI lengkap untuk satu jadwal berdasarkan order_id.
    """
    data = _schedule_cache.get(order_id.upper())
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Jadwal '{order_id}' tidak ditemukan"
        )

    kpi = data["kpi"]
    return {
        "order_id"              : order_id.upper(),
        "month"                 : data["month"],
        "approval_status"       : data.get("approval_status", "pending"),
        "kpi"                   : kpi,
        "breakdown_by_machine"  : _breakdown_by_machine(data["schedule"]),
        "breakdown_by_product"  : _breakdown_by_product(data["schedule"]),
        "prd_acceptance": {
            "cost_per_mt"       : {
                "value"  : kpi["kwh_per_mt"],
                "target" : kpi.get("prd_kwh_per_mt_target", 140),
                "met"    : kpi.get("prd_cost_target_met", kpi["kwh_per_mt"] < 140),
            },
            "operating_days"    : {
                "value"  : kpi["operating_days"],
                "manual" : kpi.get("manual_days_estimate", 22),
                "max_allowed": round(kpi.get("manual_days_estimate", 22) * 0.70, 1),
                "reduction_pct": kpi.get("days_reduction_pct", 0),
                "met"    : kpi.get("prd_days_target_met", False),
            },
            "overall"           : "PASS" if kpi["prd_kpi_target_met"] else "FAIL",
        },
    }


def _breakdown_by_machine(schedule: list) -> list:
    """Agregasi kWh dan jam operasi per mesin dari jadwal."""
    agg = {}
    for row in schedule:
        mid = row["machine_id"]
        if mid not in agg:
            agg[mid] = {
                "machine_id"    : mid,
                "machine_label" : row["machine_label"],
                "machine_type"  : row["machine_type"],
                "total_kwh"     : 0.0,
                "total_hours"   : 0.0,
                "n_sessions"    : 0,
            }
        agg[mid]["total_kwh"]   += row["predicted_kwh"]
        agg[mid]["total_hours"] += row["operating_hours"]
        agg[mid]["n_sessions"]  += 1

    for v in agg.values():
        v["total_kwh"]   = round(v["total_kwh"],   2)
        v["total_hours"] = round(v["total_hours"],  1)

    return sorted(agg.values(), key=lambda x: x["total_kwh"], reverse=True)


def _breakdown_by_product(schedule: list) -> list:
    """Agregasi kWh dan MT per produk dari jadwal."""
    agg = {}
    for row in schedule:
        pid = row["product_id"]
        if pid not in agg:
            agg[pid] = {
                "product_id"   : pid,
                "product_label": row["product_label"],
                "total_kwh"    : 0.0,
                "quantity_mt"  : row["quantity_mt"],
                "n_steps"      : 0,
            }
        agg[pid]["total_kwh"] += row["predicted_kwh"]
        agg[pid]["n_steps"]   += 1

    for v in agg.values():
        v["total_kwh"]  = round(v["total_kwh"], 2)
        v["kwh_per_mt"] = round(v["total_kwh"] / v["quantity_mt"], 4) if v["quantity_mt"] > 0 else 0

    return sorted(agg.values(), key=lambda x: x["total_kwh"], reverse=True)
