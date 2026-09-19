"""
Pydantic Schemas — Request & Response
======================================
Mendefinisikan struktur data yang masuk dan keluar dari setiap endpoint.
FastAPI menggunakan ini untuk validasi otomatis dan dokumentasi Swagger.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


# ── REQUEST ───────────────────────────────────────────────────────────────────

class OrderItem(BaseModel):
    """Satu item order dari Supply Chain."""
    product_id  : str   = Field(..., example="P004",
                                description="ID produk (P001–P007)")
    quantity_mt : float = Field(..., gt=0, le=10000, example=100.0,
                                description="Kuantitas dalam Metric Ton (> 0)")

    @field_validator("product_id")
    @classmethod
    def validate_product(cls, v):
        allowed = {"P001", "P002", "P003", "P004", "P005", "P006", "P007"}
        if v not in allowed:
            raise ValueError(f"product_id '{v}' tidak dikenal. Pilihan: {allowed}")
        return v


class ScheduleRequest(BaseModel):
    """Body request untuk POST /api/predict-schedule."""
    month  : str        = Field(
        ..., pattern=r"^\d{4}-\d{2}$",
        example="2025-11",
        description="Bulan target produksi (format: YYYY-MM)"
    )
    orders : List[OrderItem] = Field(
        ..., min_length=1, max_length=20,
        description="List order dari Supply Chain (minimal 1 produk)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "month" : "2025-11",
                "orders": [
                    {"product_id": "P001", "quantity_mt": 20},
                    {"product_id": "P004", "quantity_mt": 100},
                    {"product_id": "P006", "quantity_mt": 150},
                ],
            }
        }
    }


# ── RESPONSE ──────────────────────────────────────────────────────────────────

class KpiResponse(BaseModel):
    """Ringkasan KPI hasil optimasi (PRD §6.4)."""
    total_kwh            : float
    total_mt             : float
    kwh_per_mt           : float
    operating_days       : int
    manual_kwh_estimate  : float
    manual_days_estimate : int = 22
    saving_kwh           : float
    saving_pct           : float
    saving_rp            : float = 0
    days_saved           : int
    days_reduction_pct   : float = 0
    prd_kwh_per_mt_target: float = 140.0
    prd_cost_target_met  : bool = False
    prd_days_target_met  : bool = False
    prd_kpi_target_met   : bool
    tariff_rp_per_kwh    : float = 1525.0


class SessionRow(BaseModel):
    """Satu baris jadwal mesin (satu step BOM dari satu order)."""
    session_id      : str
    po_ref          : str
    product_id      : str
    product_label   : str
    route_step      : int
    machine_id      : str
    machine_label   : str
    machine_type    : str
    process_desc    : str
    quantity_mt     : float
    shift           : str
    operating_hours : float
    start_date      : str
    end_date        : str
    predicted_kwh   : float
    kwh_per_mt      : float
    instruction     : str
    wip_note        : Optional[str]
    status          : str


class StandbyMachine(BaseModel):
    """Mesin tidak dioperasikan + alasan (PRD §6.3)."""
    machine_id   : str
    machine_label: str
    machine_type : str
    reason       : str


class ScheduleResponse(BaseModel):
    """Response lengkap dari POST /api/predict-schedule."""
    order_id         : Optional[str] = None
    month            : str
    generated_at     : str
    approval_status  : str = "pending"
    kpi              : KpiResponse
    schedule         : List[SessionRow]
    machines_standby : List[StandbyMachine]
    total_sessions   : int


# ── PRODUCTS & MACHINES ───────────────────────────────────────────────────────

class RouteStep(BaseModel):
    step         : int
    machine_type : str
    hours_per_mt : float
    desc         : str


class ProductInfo(BaseModel):
    product_id    : str
    label         : str
    energy_factor : float
    n_steps       : int
    route         : List[RouteStep]


class MachineInfo(BaseModel):
    machine_id    : str
    label         : Optional[str]
    type          : str
    base_power_kw : int
    efficiency    : float
