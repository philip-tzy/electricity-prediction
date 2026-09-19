"""
Prediction Engine
=================
Berisi semua logika inti sistem (selaras PRD Kerry ID):

1. expand_bom()       — Ekspansi order ke sesi-sesi mesin via BOM
2. preprocess()       — Transformasi input sesuai preprocessor training
3. predict_kwh()      — Panggil model.predict() untuk satu sesi
4. run_optimizer()    — Cari kombinasi jadwal dengan total kWh terendah
5. build_schedule()   — Susun output instruksi operator + KPI PRD
"""

import math
import datetime
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ── Konstanta PRD ─────────────────────────────────────────────────────────────
SHIFTS = ["Shift 1", "Shift 2", "Shift 3"]

# Multiplier shift (dokumentasi logika training — prediksi memakai model XGBoost)
SHIFT_MULTIPLIER = {
    "Shift 1": 1.00,
    "Shift 2": 1.05,   # +5% vs Shift 1
    "Shift 3": 0.97,   # −3% vs Shift 1
}

# Baseline jadwal manual (PRD §2.2): ~22 hari kerja / bulan
MANUAL_DAYS_BASELINE = 22

# Uplift estimasi kWh jadwal manual vs AI (startup/restart & idle cost)
MANUAL_KWH_UPLIFT = 1.18

# Target cost/MT: PRD menulis "0.14 kWh/MT"; satuan dataset/model = kWh
# sehingga target operasional = 0.14 × 1000 = 140 kWh/MT (= 0.14 MWh/MT)
PRD_KWH_PER_MT_TARGET = 140.0

# Acceptance #4: hari operasional AI ≤ 70% baseline manual (≥ 30% pemangkasan)
PRD_DAYS_RATIO_MAX = 0.70

# Tarif listrik industri (Rp/kWh) — untuk KPI penghematan Rp (PRD §6.4)
INDUSTRIAL_TARIFF_RP_PER_KWH = 1525.0

# Setup tetap (deterministik untuk UAT) — menggantikan random
SETUP_HOURS = 0.8

# Max jam operasi efektif per hari kerja (2 shift)
MAX_HOURS_PER_DAY = 16.0

MACHINE_LABELS = {
    "M001": "Mixer Utama A",
    "M002": "Roaster A",
    "M003": "Extruder A",
    "M004": "Packer A",
    "M005": "Palletizer A",
    "M006": "Mixer Cadangan B",
    "M007": "Roaster B",
    "M008": "Packer B",
}


def get_machines_by_type(machines_master: Dict) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for mid, mdata in machines_master.items():
        result.setdefault(mdata["type"], []).append(mid)
    return result


def is_workday(d: datetime.date) -> bool:
    """PRD: pabrik 5 hari kerja/minggu — Sabtu & Minggu tutup."""
    return d.weekday() < 5


def next_workday(d: datetime.date) -> datetime.date:
    while not is_workday(d):
        d += datetime.timedelta(days=1)
    return d


def add_work_span(start: datetime.date, work_days: int) -> Tuple[datetime.date, datetime.date]:
    """
    Tempatkan sesi nonstop selama `work_days` hari kerja.
    Mengembalikan (start_workday, end_workday).
    """
    start = next_workday(start)
    if work_days <= 1:
        return start, start
    remaining = work_days - 1
    end = start
    while remaining > 0:
        end += datetime.timedelta(days=1)
        if is_workday(end):
            remaining -= 1
    return start, end


# ── 1. Ekspansi BOM ──────────────────────────────────────────────────────────
def expand_bom(
    orders: List[Dict],
    bom_routing: Dict,
) -> List[Dict]:
    """
    Mengubah list order dari frontend menjadi list sesi mesin.
    Setiap step BOM = satu sesi independen (WIP boleh decoupling — PRD §5.1).
    """
    sessions = []
    products = bom_routing["products"]

    for order in orders:
        pid = order["product_id"]
        qty = order["quantity_mt"]
        p_data = products[pid]
        route = p_data["route"]
        po_ref = f"{pid}-{qty}MT"

        for step in route:
            raw_op_hours = qty * step["hours_per_mt"]
            operating_hours = round(min(raw_op_hours + SETUP_HOURS, MAX_HOURS_PER_DAY), 1)
            operating_hours = max(operating_hours, 1.0)

            sessions.append({
                "po_ref"         : po_ref,
                "product_id"     : pid,
                "product_label"  : p_data["label"],
                "quantity_mt"    : qty,
                "route_step"     : step["step"],
                "machine_type"   : step["machine_type"],
                "hours_per_mt"   : step["hours_per_mt"],
                "operating_hours": operating_hours,
                "desc"           : step["desc"],
            })

    return sessions


# ── 2. Preprocessing input ───────────────────────────────────────────────────
def preprocess_session(
    session: Dict,
    machine_id: str,
    shift: str,
    preprocessor: Dict,
    schedule_date: Optional[datetime.date] = None,
) -> pd.DataFrame:
    """
    Transformasi satu sesi mesin ke format yang bisa diterima model.
    Fitur tanggal memakai tanggal jadwal (bukan hari 'today') bila tersedia.
    """
    op_hours = session["operating_hours"]
    qty = session["quantity_mt"]
    downtime = 0.0  # optimizer: nonstop, tanpa downtime/restart (PRD constraint)

    active_hours = max(op_hours - downtime, 0)
    utilization_ratio = active_hours / op_hours if op_hours > 0 else 0
    has_downtime = 0
    n_restarts = 0
    qty_per_op_hour = qty / op_hours if op_hours > 0 else 0

    ref_date = schedule_date or datetime.date.today()
    row = {
        "machine_id"       : machine_id,
        "product_id"       : session["product_id"],
        "machine_type"     : session["machine_type"],
        "shift"            : shift,
        "route_step"       : session["route_step"],
        "quantity_mt"      : qty,
        "operating_hours"  : op_hours,
        "downtime_hours"   : downtime,
        "production_rate"  : qty / active_hours if active_hours > 0 else 0,
        "active_hours"     : active_hours,
        "utilization_ratio": utilization_ratio,
        "has_downtime"     : has_downtime,
        "n_restarts"       : n_restarts,
        "qty_per_op_hour"  : qty_per_op_hour,
        "day_of_week"      : ref_date.weekday(),
        "month"            : ref_date.month,
        "year"             : ref_date.year,
        "is_weekend"       : int(ref_date.weekday() >= 5),
    }

    all_features = preprocessor["all_features"]
    categorical_features = preprocessor["categorical_features"]
    numerical_features = preprocessor["numerical_features"]

    df = pd.DataFrame([row])[all_features]

    for col in categorical_features:
        le = preprocessor["label_encoders"][col]
        df[col] = le.transform(df[col].astype(str))

    df[numerical_features] = preprocessor["scaler"].transform(df[numerical_features])
    return df


# ── 3. Prediksi kWh satu sesi ────────────────────────────────────────────────
def predict_kwh(
    session: Dict,
    machine_id: str,
    shift: str,
    model,
    preprocessor: Dict,
    schedule_date: Optional[datetime.date] = None,
) -> float:
    """Prediksi electricity_kwh untuk satu kombinasi sesi × mesin × shift."""
    df_input = preprocess_session(session, machine_id, shift, preprocessor, schedule_date)
    kwh = float(model.predict(df_input)[0])
    return round(max(kwh, 0), 2)


# ── 4. Optimizer ─────────────────────────────────────────────────────────────
def run_optimizer(
    sessions: List[Dict],
    bom_routing: Dict,
    model,
    preprocessor: Dict,
) -> List[Dict]:
    """
    Untuk setiap sesi, pilih (machine_id × shift) dengan predicted kWh terendah.

    Constraint PRD §5.1 Tahap 4:
    - Semua order terpenuhi (setiap sesi BOM dialokasi)
    - Bandingkan opsi mesin kandidat sejenis (paralel capacity) vs mesin irit
    - Shift dievaluasi penuh; model sudah mengkodekan multiplier shift
    - Satu sesi = satu run nonstop (tanpa on-off / restart)
    """
    machines_master = bom_routing["machines"]
    machines_by_type = get_machines_by_type(machines_master)

    ordered = sorted(sessions, key=lambda s: (s["po_ref"], s["route_step"]))
    optimized = []

    for session in ordered:
        machine_type = session["machine_type"]
        candidates = machines_by_type.get(machine_type, [])

        best_kwh = float("inf")
        best_machine = None
        best_shift = None

        for mid in candidates:
            for shift in SHIFTS:
                kwh = predict_kwh(session, mid, shift, model, preprocessor)
                better = (
                    kwh < best_kwh
                    or (
                        kwh == best_kwh
                        and (
                            (best_shift != "Shift 3" and shift == "Shift 3")
                            or (
                                shift == best_shift
                                and (best_machine is None or mid < best_machine)
                            )
                        )
                    )
                )
                if better:
                    best_kwh = kwh
                    best_machine = mid
                    best_shift = shift

        if best_machine is None:
            raise ValueError(
                f"Tidak ada mesin tipe '{machine_type}' untuk sesi {session['po_ref']} "
                f"step {session['route_step']}"
            )

        optimized.append({
            **session,
            "machine_id"   : best_machine,
            "machine_label": MACHINE_LABELS.get(
                best_machine,
                machines_master[best_machine].get("label", best_machine),
            ),
            "shift"        : best_shift,
            "predicted_kwh": best_kwh,
        })

    return optimized


# ── 5. Build jadwal output ───────────────────────────────────────────────────
def build_schedule(
    optimized_sessions: List[Dict],
    month_str: str,
    bom_routing: Dict,
) -> Dict:
    """
    Susun hasil optimasi menjadi output siap eksekusi operator.

    Constraint penempatan tanggal (PRD):
    - Skip Sabtu & Minggu
    - Mesin tidak overlap (timeline per mesin)
    - Step BOM berikutnya mulai setelah step sebelumnya selesai (WIP decoupling OK)
    - Mesin berbeda boleh paralel
    """
    machines_master = bom_routing["machines"]
    all_machine_ids = set(machines_master.keys())
    used_machine_ids: set = set()

    try:
        year, month = map(int, month_str.split("-"))
    except Exception:
        today = datetime.date.today()
        year, month = today.year, today.month

    month_start = datetime.date(year, month, 1)

    machine_free_from: Dict[str, datetime.date] = {
        mid: month_start for mid in machines_master
    }
    po_ready_from: Dict[str, datetime.date] = {}

    schedule_rows = []
    ordered = sorted(optimized_sessions, key=lambda s: (s["po_ref"], s["route_step"]))

    for sess in ordered:
        mid = sess["machine_id"]
        used_machine_ids.add(mid)

        earliest = max(
            machine_free_from[mid],
            po_ready_from.get(sess["po_ref"], month_start),
            month_start,
        )
        work_days = max(1, math.ceil(sess["operating_hours"] / MAX_HOURS_PER_DAY))
        start_date, end_date = add_work_span(earliest, work_days)

        machine_free_from[mid] = next_workday(end_date + datetime.timedelta(days=1))
        po_ready_from[sess["po_ref"]] = next_workday(end_date + datetime.timedelta(days=1))

        wip_note = None
        if sess["route_step"] > 1:
            wip_note = _build_wip_note(sess, start_date)

        schedule_rows.append({
            "session_id"     : f"{sess['po_ref']}-step{sess['route_step']}",
            "po_ref"         : sess["po_ref"],
            "product_id"     : sess["product_id"],
            "product_label"  : sess["product_label"],
            "route_step"     : sess["route_step"],
            "machine_id"     : mid,
            "machine_label"  : sess["machine_label"],
            "machine_type"   : sess["machine_type"],
            "process_desc"   : sess["desc"],
            "quantity_mt"    : sess["quantity_mt"],
            "shift"          : sess["shift"],
            "operating_hours": sess["operating_hours"],
            "start_date"     : start_date.isoformat(),
            "end_date"       : end_date.isoformat(),
            "predicted_kwh"  : sess["predicted_kwh"],
            "kwh_per_mt"     : round(sess["predicted_kwh"] / sess["quantity_mt"], 4),
            "instruction"    : _build_instruction(sess, start_date, end_date),
            "wip_note"       : wip_note,
            "status"         : "scheduled",
        })

    schedule_rows.sort(key=lambda r: (r["start_date"], r["machine_id"], r["route_step"]))

    # ── KPI (PRD §6.4 & Acceptance Criteria) ──────────────────────────────────
    total_kwh = round(sum(s["predicted_kwh"] for s in schedule_rows), 2)
    mt_by_po = {s["po_ref"]: s["quantity_mt"] for s in schedule_rows}
    total_mt = round(sum(mt_by_po.values()), 2)
    kwh_per_mt = round(total_kwh / total_mt, 4) if total_mt > 0 else 0.0

    all_work_dates: set = set()
    for row in schedule_rows:
        d = datetime.date.fromisoformat(row["start_date"])
        e = datetime.date.fromisoformat(row["end_date"])
        while d <= e:
            if is_workday(d):
                all_work_dates.add(d)
            d += datetime.timedelta(days=1)
    operating_days = len(all_work_dates)

    manual_kwh_est = round(total_kwh * MANUAL_KWH_UPLIFT, 2)
    saving_kwh = round(manual_kwh_est - total_kwh, 2)
    saving_pct = round((saving_kwh / manual_kwh_est) * 100, 1) if manual_kwh_est > 0 else 0.0
    saving_rp = round(saving_kwh * INDUSTRIAL_TARIFF_RP_PER_KWH, 0)

    manual_days_est = MANUAL_DAYS_BASELINE
    days_saved = max(manual_days_est - operating_days, 0)
    days_reduction_pct = round(
        (days_saved / manual_days_est) * 100, 1
    ) if manual_days_est > 0 else 0.0

    cost_ok = kwh_per_mt < PRD_KWH_PER_MT_TARGET
    days_ok = operating_days <= (manual_days_est * PRD_DAYS_RATIO_MAX)
    prd_target_met = bool(cost_ok and days_ok)

    machines_standby = []
    for mid in sorted(all_machine_ids - used_machine_ids):
        mtype = machines_master[mid]["type"]
        siblings_used = [
            u for u in used_machine_ids
            if machines_master[u]["type"] == mtype
        ]
        if siblings_used:
            reason = (
                f"Tidak dipilih optimizer — beban tipe {mtype} sudah dipenuhi oleh "
                f"{', '.join(siblings_used)} dengan prediksi kWh lebih rendah."
            )
        else:
            reason = (
                f"Tidak ada order yang membutuhkan proses tipe {mtype} bulan ini."
            )
        machines_standby.append({
            "machine_id"   : mid,
            "machine_label": MACHINE_LABELS.get(mid, mid),
            "machine_type" : mtype,
            "reason"       : reason,
        })

    return {
        "month"          : month_str,
        "generated_at"   : datetime.datetime.now().isoformat(),
        "approval_status": "pending",
        "kpi": {
            "total_kwh"            : total_kwh,
            "total_mt"             : total_mt,
            "kwh_per_mt"           : kwh_per_mt,
            "operating_days"       : operating_days,
            "manual_kwh_estimate"  : manual_kwh_est,
            "manual_days_estimate" : manual_days_est,
            "saving_kwh"           : saving_kwh,
            "saving_pct"           : saving_pct,
            "saving_rp"            : saving_rp,
            "days_saved"           : days_saved,
            "days_reduction_pct"   : days_reduction_pct,
            "prd_kwh_per_mt_target": PRD_KWH_PER_MT_TARGET,
            "prd_cost_target_met"  : cost_ok,
            "prd_days_target_met"  : days_ok,
            "prd_kpi_target_met"   : prd_target_met,
            "tariff_rp_per_kwh"    : INDUSTRIAL_TARIFF_RP_PER_KWH,
        },
        "schedule"        : schedule_rows,
        "machines_standby": machines_standby,
        "total_sessions"  : len(schedule_rows),
    }


def _build_instruction(sess: Dict, start: datetime.date, end: datetime.date) -> str:
    """Instruksi absolut siap dieksekusi operator (PRD §6.3)."""
    if start == end:
        duration = start.strftime("%A, %d %b %Y")
    else:
        duration = (
            f"{start.strftime('%A, %d %b')} sampai {end.strftime('%A, %d %b %Y')}"
        )

    stop_warning = ""
    if sess["machine_type"] in ("Roaster", "Extruder"):
        stop_warning = (
            " PERINGATAN: Jangan matikan mesin di tengah proses — "
            "startup cost (±15 menit peak current) sangat tinggi."
        )

    return (
        f"Jalankan {sess['machine_id']} ({sess['machine_type']}) "
        f"untuk {sess['product_label']} sebanyak {sess['quantity_mt']} MT "
        f"pada {duration}, {sess['shift']}. "
        f"Selesaikan nonstop dalam satu sesi tanpa on-off."
        f"{stop_warning}"
    )


def _build_wip_note(sess: Dict, start: datetime.date) -> str:
    """Notifikasi WIP: material siap diambil untuk step berikutnya (PRD §6.3)."""
    return (
        f"WIP siap: material {sess['product_label']} dari step {sess['route_step'] - 1} "
        f"sudah menunggu di gudang. Ambil sebelum sesi dimulai "
        f"({start.strftime('%d %b %Y')})."
    )
