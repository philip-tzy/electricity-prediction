"""
Prediction Engine
=================
Berisi semua logika inti sistem (selaras PRD Kerry ID):

1. expand_bom()       — Ekspansi order → sesi 8 jam via BOM + production_rate
2. preprocess()       — Transformasi input sesuai preprocessor training
3. predict_kwh()      — Panggil model.predict() untuk satu sesi (1 shift)
4. run_optimizer()    — Pilih mesin×shift paling hemat; cadangan hanya jika perlu
5. build_schedule()   — Susun kalender nonstop + WIP + KPI PRD
"""

import math
import calendar
import datetime
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

# ── Konstanta PRD ─────────────────────────────────────────────────────────────
SHIFTS = ["Shift 1", "Shift 2", "Shift 3"]

SHIFT_MULTIPLIER = {
    "Shift 1": 1.00,
    "Shift 2": 1.05,   # +5% vs Shift 1
    "Shift 3": 0.97,   # −3% vs Shift 1
}

MANUAL_DAYS_BASELINE = 22
MANUAL_KWH_UPLIFT = 1.18
PRD_KWH_PER_MT_TARGET = 140.0
PRD_DAYS_RATIO_MAX = 0.70
INDUSTRIAL_TARIFF_RP_PER_KWH = 1525.0

# 1 sesi operasi = 1 shift 8 jam (selaras training model & skenario PRD)
MODEL_SHIFT_HOURS = 8.0

# Max 2 shift efektif per hari kerja (kapasitas kalender mesin)
MAX_SHIFTS_PER_DAY = 2
MAX_HOURS_PER_DAY = MODEL_SHIFT_HOURS * MAX_SHIFTS_PER_DAY

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


def count_workdays_in_month(year: int, month: int) -> int:
    n_days = calendar.monthrange(year, month)[1]
    return sum(
        1
        for d in range(1, n_days + 1)
        if datetime.date(year, month, d).weekday() < 5
    )


# ── 1. Ekspansi BOM → sesi per shift ─────────────────────────────────────────
def expand_bom(
    orders: List[Dict],
    bom_routing: Dict,
) -> List[Dict]:
    """
    Urai order menjadi sesi mesin berukuran 1 shift (8 jam).

    production_rate (MT/jam) = 1 / hours_per_mt
    total_hours = quantity_mt / production_rate = quantity_mt * hours_per_mt
    n_sesi = ceil(total_hours / 8)

    Contoh PRD: P004 Roaster 160 MT, ~2–2.9 MT/jam → ~56–80 jam → 7–10 sesi × 8 jam.
    """
    sessions: List[Dict] = []
    products = bom_routing["products"]

    for order_idx, order in enumerate(orders, start=1):
        pid = order["product_id"]
        qty = float(order["quantity_mt"])
        p_data = products[pid]
        route = p_data["route"]
        # po_ref HARUS unik per baris order (qty sama ≠ PO sama)
        qty_label = int(qty) if qty == int(qty) else qty
        po_ref = f"{pid}-{qty_label}MT-O{order_idx}"

        for step in route:
            hpm = float(step["hours_per_mt"])
            production_rate = 1.0 / hpm  # MT per jam untuk produk×step ini
            total_hours = qty * hpm

            chunks: List[Tuple[float, float]] = []  # (hours, qty_session)
            remaining_h = total_hours
            remaining_q = qty

            while remaining_h > 1e-9:
                hours = min(MODEL_SHIFT_HOURS, remaining_h)
                hours = round(hours, 2)
                if hours < 0.25 and chunks:
                    prev_h, prev_q = chunks[-1]
                    chunks[-1] = (round(prev_h + hours, 2), round(prev_q + remaining_q, 3))
                    break

                if remaining_h - hours <= 1e-9:
                    q_sess = round(remaining_q, 3)
                else:
                    q_sess = round(production_rate * hours, 3)

                chunks.append((hours, q_sess))
                remaining_h = round(remaining_h - hours, 6)
                remaining_q = round(remaining_q - q_sess, 6)

            n_sess = len(chunks)
            for idx, (hours, q_sess) in enumerate(chunks, start=1):
                sessions.append({
                    "po_ref"            : po_ref,
                    "product_id"        : pid,
                    "product_label"     : p_data["label"],
                    "order_quantity_mt" : qty,
                    "quantity_mt"       : q_sess,
                    "route_step"        : step["step"],
                    "session_index"     : idx,
                    "session_count"     : n_sess,
                    "machine_type"      : step["machine_type"],
                    "hours_per_mt"      : hpm,
                    "production_rate"   : round(production_rate, 4),
                    "operating_hours"   : hours,
                    "desc"              : step["desc"],
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
    Transformasi satu sesi (≤8 jam) ke fitur model.
    production_rate diambil dari BOM (kapasitas MT/jam), bukan qty/hours ad-hoc.
    """
    qty = float(session["quantity_mt"])
    production_rate = float(
        session.get("production_rate")
        or (1.0 / session["hours_per_mt"] if session.get("hours_per_mt") else 0.0)
    )

    # Frame training: operating_hours selalu 8
    op_hours = MODEL_SHIFT_HOURS
    downtime = 0.0
    active_hours = MODEL_SHIFT_HOURS
    utilization_ratio = 1.0
    has_downtime = 0
    n_restarts = 0
    qty_per_shift = qty
    denom = production_rate * op_hours
    load_factor_proxy = (qty / denom) if denom > 0 else 0.0

    ref_date = schedule_date or datetime.date.today()
    row = {
        "machine_id"       : machine_id,
        "product_id"       : session["product_id"],
        "machine_type"     : session["machine_type"],
        "shift"            : shift,
        "route_step"       : session["route_step"],
        "quantity_mt"      : qty_per_shift,
        "operating_hours"  : op_hours,
        "downtime_hours"   : downtime,
        "production_rate"  : production_rate,
        "active_hours"     : active_hours,
        "utilization_ratio": utilization_ratio,
        "has_downtime"     : has_downtime,
        "n_restarts"       : n_restarts,
        "load_factor_proxy": load_factor_proxy,
        "qty_per_shift"    : qty_per_shift,
        "day_of_week"      : ref_date.weekday(),
        "month"            : ref_date.month,
        "year"             : ref_date.year,
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
    """Prediksi electricity_kwh untuk satu sesi (satu shift)."""
    df_input = preprocess_session(session, machine_id, shift, preprocessor, schedule_date)
    kwh = float(model.predict(df_input)[0])

    # Sesi parsial (< 8 jam): skala proporsional
    op_hours = float(session["operating_hours"])
    if op_hours < MODEL_SHIFT_HOURS:
        kwh = kwh * (op_hours / MODEL_SHIFT_HOURS)

    if kwh <= 0:
        kwh = max(op_hours * 50.0, float(session["quantity_mt"]) * 4.0)
    return round(kwh, 2)


# ── 4. Optimizer ─────────────────────────────────────────────────────────────
def run_optimizer(
    sessions: List[Dict],
    bom_routing: Dict,
    model,
    preprocessor: Dict,
    month_str: Optional[str] = None,
) -> List[Dict]:
    """
    Alokasi mesin×shift:

    1. Satu grup (PO × step) tetap di satu mesin/shift bila muat — nonstop, anti-restart.
    2. Pilih mesin dengan skor = prediksi_kWh + penalti beban antrean
       → PO berbeda otomatis tersebar ke M001/M006, M002/M007, M004/M008.
    3. Jika satu grup lebih besar dari sisa kapasitas mesin, pecah overflow ke cadangan.
    """
    machines_master = bom_routing["machines"]
    machines_by_type = get_machines_by_type(machines_master)

    workdays_budget = MANUAL_DAYS_BASELINE
    if month_str:
        try:
            year, month = map(int, month_str.split("-"))
            workdays_budget = count_workdays_in_month(year, month)
        except Exception:
            pass

    groups: "OrderedDict[Tuple[str, int], List[Dict]]" = OrderedDict()
    for s in sessions:
        key = (s["po_ref"], s["route_step"])
        groups.setdefault(key, []).append(s)

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x["session_index"])

    group_keys = sorted(
        groups.keys(),
        key=lambda k: (
            groups[k][0]["route_step"],
            -sum(x["operating_hours"] for x in groups[k]),
            k[0],
        ),
    )

    machine_slots: Dict[str, int] = {mid: 0 for mid in machines_master}
    slot_budget = max(workdays_budget * MAX_SHIFTS_PER_DAY, 1)
    # Penalti per slot terisi — dorong sebaran antar mesin paralel
    LOAD_PENALTY = 25.0
    optimized: List[Dict] = []

    def best_shift_for(mid: str, sample: Dict) -> Tuple[float, str]:
        best = (float("inf"), SHIFTS[0])
        for shift in SHIFTS:
            kwh = predict_kwh(sample, mid, shift, model, preprocessor)
            if kwh < best[0]:
                best = (kwh, shift)
        return best

    for key in group_keys:
        group = groups[key]
        sample = group[0]
        mtype = sample["machine_type"]
        candidates = machines_by_type.get(mtype, [])
        if not candidates:
            raise ValueError(
                f"Tidak ada mesin tipe '{mtype}' untuk {key[0]} step {key[1]}"
            )

        remaining = list(group)

        while remaining:
            # Skor tiap mesin: energi + beban saat ini
            scored: List[Tuple[float, float, str, str]] = []
            for mid in candidates:
                kwh, shift = best_shift_for(mid, remaining[0])
                score = kwh + LOAD_PENALTY * machine_slots[mid]
                scored.append((score, kwh, mid, shift))
            scored.sort(key=lambda x: (x[0], x[1], x[2]))

            placed = False
            for _, _, mid, shift in scored:
                slots_left = max(slot_budget - machine_slots[mid], 0)
                if slots_left <= 0:
                    continue
                take_n = min(len(remaining), slots_left)
                batch = remaining[:take_n]
                remaining = remaining[take_n:]
                for s in batch:
                    kwh = predict_kwh(s, mid, shift, model, preprocessor)
                    machine_slots[mid] += 1
                    optimized.append({
                        **s,
                        "machine_id"   : mid,
                        "machine_label": MACHINE_LABELS.get(
                            mid,
                            machines_master[mid].get("label", mid),
                        ),
                        "shift"        : shift,
                        "predicted_kwh": kwh,
                    })
                placed = True
                break

            if not placed:
                # Semua mesin "penuh" vs budget bulan — tetap alokasi ke paling longgar
                scored_by_load = sorted(
                    scored, key=lambda x: (machine_slots[x[2]], x[1], x[2])
                )
                _, _, mid, shift = scored_by_load[0]
                for s in remaining:
                    kwh = predict_kwh(s, mid, shift, model, preprocessor)
                    machine_slots[mid] += 1
                    optimized.append({
                        **s,
                        "machine_id"   : mid,
                        "machine_label": MACHINE_LABELS.get(
                            mid,
                            machines_master[mid].get("label", mid),
                        ),
                        "shift"        : shift,
                        "predicted_kwh": kwh,
                    })
                remaining = []

    return optimized


# ── 5. Build jadwal output ───────────────────────────────────────────────────
def build_schedule(
    optimized_sessions: List[Dict],
    month_str: str,
    bom_routing: Dict,
) -> Dict:
    """
    Tempatkan setiap sesi 8 jam ke kalender kerja (event-based).

    - Kapasitas mesin: max 2 sesi/hari kerja (16 jam) — nonstop day+night
    - Shift label = profil energi prediksi; packing kalender memakai kapasitas fisik
    - Step berikutnya hanya setelah SEMUA sesi step sebelumnya selesai (WIP)
    - Sesi dalam step yang sama berurutan (nonstop)
    - PO berbeda boleh paralel di mesin berbeda
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

    # (machine_id, date) -> jumlah sesi yang sudah ditempatkan hari itu
    day_slots: Dict[Tuple[str, datetime.date], int] = {}

    def sess_key(s: Dict) -> Tuple:
        return (s["po_ref"], s["route_step"], s["session_index"])

    unscheduled: Dict[Tuple, Dict] = {sess_key(s): s for s in optimized_sessions}
    done_keys: set = set()
    step_done_count: Dict[Tuple[str, int], int] = {}
    step_complete: Dict[Tuple[str, int], bool] = {}
    step_end_date: Dict[Tuple[str, int], datetime.date] = {}

    schedule_rows: List[Dict] = []

    def is_ready(s: Dict) -> bool:
        if s["route_step"] > 1:
            if not step_complete.get((s["po_ref"], s["route_step"] - 1), False):
                return False
        if s["session_index"] > 1:
            prev = (s["po_ref"], s["route_step"], s["session_index"] - 1)
            if prev not in done_keys:
                return False
        return True

    def find_slot(mid: str, earliest: datetime.date) -> datetime.date:
        """Cari hari kerja pertama >= earliest dengan sisa kapasitas slot mesin."""
        d = next_workday(earliest)
        while True:
            used = day_slots.get((mid, d), 0)
            if used < MAX_SHIFTS_PER_DAY:
                return d
            d = next_workday(d + datetime.timedelta(days=1))

    def earliest_for(s: Dict) -> datetime.date:
        mid = s["machine_id"]
        t = month_start
        if s["route_step"] > 1:
            prev_end = step_end_date.get((s["po_ref"], s["route_step"] - 1))
            if prev_end is not None:
                t = max(t, next_workday(prev_end + datetime.timedelta(days=1)))
        # Sesi sebelumnya di step yang sama: boleh hari yang sama jika slot masih ada,
        # atau hari berikutnya — find_slot menangani kapasitas.
        if s["session_index"] > 1:
            prev_key = (s["po_ref"], s["route_step"], s["session_index"] - 1)
            # cari end date sesi sebelumnya dari schedule_rows
            for row in reversed(schedule_rows):
                if (
                    row["po_ref"] == s["po_ref"]
                    and row["route_step"] == s["route_step"]
                    and row["session_id"].endswith(f"-s{s['session_index']-1:02d}")
                ):
                    prev_day = datetime.date.fromisoformat(row["end_date"])
                    # lanjut nonstop: coba slot di hari yang sama dulu
                    t = max(t, prev_day)
                    break
        return find_slot(mid, t)

    while unscheduled:
        ready = [s for s in unscheduled.values() if is_ready(s)]
        if not ready:
            ready = list(unscheduled.values())

        ready.sort(
            key=lambda s: (
                earliest_for(s),
                s["route_step"],
                -s["operating_hours"],
                s["po_ref"],
                s["session_index"],
            )
        )
        sess = ready[0]
        mid = sess["machine_id"]
        used_machine_ids.add(mid)

        start_date = earliest_for(sess)
        end_date = start_date
        day_slots[(mid, start_date)] = day_slots.get((mid, start_date), 0) + 1

        key = sess_key(sess)
        done_keys.add(key)
        del unscheduled[key]

        sk = (sess["po_ref"], sess["route_step"])
        step_done_count[sk] = step_done_count.get(sk, 0) + 1
        step_end_date[sk] = end_date
        if step_done_count[sk] >= sess["session_count"]:
            step_complete[sk] = True

        wip_note = None
        if sess["route_step"] > 1 and sess["session_index"] == 1:
            wip_note = _build_wip_note(sess, start_date)

        order_qty = float(sess.get("order_quantity_mt", sess["quantity_mt"]))
        schedule_rows.append({
            "session_id"     : (
                f"{sess['po_ref']}-step{sess['route_step']}"
                f"-s{sess['session_index']:02d}"
            ),
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
            "kwh_per_mt"     : round(
                sess["predicted_kwh"] / sess["quantity_mt"], 4
            ) if sess["quantity_mt"] else 0.0,
            "instruction"    : _build_instruction(sess, start_date, end_date, order_qty),
            "wip_note"       : wip_note,
            "status"         : "scheduled",
        })

    schedule_rows.sort(key=lambda r: (r["start_date"], r["machine_id"], r["route_step"]))
    _enrich_block_instructions(schedule_rows)

    # ── KPI ───────────────────────────────────────────────────────────────────
    total_kwh = round(sum(s["predicted_kwh"] for s in schedule_rows), 2)
    mt_by_po: Dict[str, float] = {}
    for o in optimized_sessions:
        mt_by_po[o["po_ref"]] = float(o.get("order_quantity_mt", o["quantity_mt"]))
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

    # Baseline manual scaling: order besar butuh lebih dari 22 hari
    month_workdays = count_workdays_in_month(year, month)
    manual_days_est = max(
        MANUAL_DAYS_BASELINE,
        month_workdays,
        int(math.ceil(operating_days * MANUAL_KWH_UPLIFT)),
    )
    days_saved = max(manual_days_est - operating_days, 0)
    days_reduction_pct = round(
        (days_saved / manual_days_est) * 100, 1
    ) if manual_days_est > 0 else 0.0

    cost_ok = kwh_per_mt < PRD_KWH_PER_MT_TARGET
    days_ok = operating_days <= max(int(manual_days_est * PRD_DAYS_RATIO_MAX), month_workdays)
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
                f"Standby — kapasitas tipe {mtype} sudah dipenuhi oleh "
                f"{', '.join(siblings_used)} dengan prediksi kWh lebih rendah. "
                f"Siap sebagai backup breakdown / bulan depan."
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


def _enrich_block_instructions(schedule_rows: List[Dict]) -> None:
    """Instruksi nonstop untuk blok sesi berurutan (PO+step+mesin+shift)."""
    blocks: Dict[Tuple, List[Dict]] = OrderedDict()
    for row in schedule_rows:
        key = (row["po_ref"], row["route_step"], row["machine_id"], row["shift"])
        blocks.setdefault(key, []).append(row)

    for rows in blocks.values():
        if len(rows) <= 1:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["start_date"])
        start = datetime.date.fromisoformat(rows_sorted[0]["start_date"])
        end = datetime.date.fromisoformat(rows_sorted[-1]["end_date"])
        mid = rows_sorted[0]["machine_id"]
        mtype = rows_sorted[0]["machine_type"]
        label = rows_sorted[0]["product_label"]
        shift = rows_sorted[0]["shift"]
        n = len(rows_sorted)
        total_qty = round(sum(r["quantity_mt"] for r in rows_sorted), 1)

        stop_warning = ""
        if mtype in ("Roaster", "Extruder"):
            stop_warning = (
                " Jangan matikan / restart di tengah proses — "
                "startup cost peak current sangat tinggi."
            )

        block_instr = (
            f"Jalankan {mid} ({mtype}) untuk {label} sebanyak {total_qty} MT "
            f"nonstop {n} sesi x {MODEL_SHIFT_HOURS:.0f} jam, "
            f"{start.strftime('%A, %d %b %Y')} sampai {end.strftime('%A, %d %b %Y')}, "
            f"{shift}.{stop_warning}"
        )
        for r in rows_sorted:
            r["instruction"] = block_instr


def _build_instruction(
    sess: Dict,
    start: datetime.date,
    end: datetime.date,
    order_qty: float,
) -> str:
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

    sess_info = ""
    if sess.get("session_count", 1) > 1:
        sess_info = (
            f" (sesi {sess['session_index']}/{sess['session_count']}, "
            f"rate {sess.get('production_rate', 0):.2f} MT/jam)"
        )

    return (
        f"Jalankan {sess['machine_id']} ({sess['machine_type']}) "
        f"untuk {sess['product_label']} {sess['quantity_mt']} MT"
        f"{sess_info} pada {duration}, {sess['shift']}. "
        f"Selesaikan nonstop tanpa on-off."
        f"{stop_warning}"
    )


def _build_wip_note(sess: Dict, start: datetime.date) -> str:
    """Notifikasi WIP: material siap diambil untuk step berikutnya (PRD §6.3)."""
    return (
        f"WIP siap: material {sess['product_label']} dari step {sess['route_step'] - 1} "
        f"sudah menunggu di gudang. Ambil sebelum sesi dimulai "
        f"({start.strftime('%d %b %Y')})."
    )
