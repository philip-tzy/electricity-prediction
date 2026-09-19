"""
Router: Machines
GET /api/machines           → Daftar semua mesin + kapasitas
GET /api/machines/{id}      → Detail satu mesin
GET /api/machines/type/{type} → Mesin berdasarkan tipe
"""

from fastapi import APIRouter, HTTPException
from typing import List

from core.model_loader import ModelLoader
from schemas.schedule import MachineInfo

router = APIRouter()

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


@router.get("/machines", response_model=List[MachineInfo], summary="Daftar semua mesin")
def get_machines():
    """
    Mengembalikan semua mesin yang terdaftar beserta kapasitas dan efisiensinya.

    Dipakai frontend untuk:
    - Menampilkan status mesin di dashboard
    - Menunjukkan mesin mana yang aktif vs standby bulan ini
    """
    _, _, bom_routing = ModelLoader.get()
    machines = bom_routing["machines"]

    return [
        MachineInfo(
            machine_id    = mid,
            label         = MACHINE_LABELS.get(mid, mid),
            type          = mdata["type"],
            base_power_kw = mdata["base_power_kw"],
            efficiency    = mdata["efficiency"],
        )
        for mid, mdata in sorted(machines.items())
    ]


@router.get("/machines/{machine_id}", response_model=MachineInfo, summary="Detail satu mesin")
def get_machine(machine_id: str):
    """Detail satu mesin berdasarkan machine_id (M001–M008)."""
    _, _, bom_routing = ModelLoader.get()
    machines = bom_routing["machines"]

    if machine_id not in machines:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' tidak ditemukan. "
                   f"Tersedia: {sorted(machines.keys())}"
        )

    mdata = machines[machine_id]
    return MachineInfo(
        machine_id    = machine_id,
        label         = MACHINE_LABELS.get(machine_id, machine_id),
        type          = mdata["type"],
        base_power_kw = mdata["base_power_kw"],
        efficiency    = mdata["efficiency"],
    )


@router.get("/machines/type/{machine_type}", response_model=List[MachineInfo],
            summary="Mesin berdasarkan tipe")
def get_machines_by_type(machine_type: str):
    """
    Mengembalikan semua mesin yang memiliki tipe tertentu.
    Tipe yang valid: Mixer, Roaster, Extruder, Packer, Palletizer
    """
    _, _, bom_routing = ModelLoader.get()
    machines = bom_routing["machines"]

    valid_types = {"Mixer", "Roaster", "Extruder", "Packer", "Palletizer"}
    if machine_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe '{machine_type}' tidak valid. Pilihan: {sorted(valid_types)}"
        )

    result = [
        MachineInfo(
            machine_id    = mid,
            label         = MACHINE_LABELS.get(mid, mid),
            type          = mdata["type"],
            base_power_kw = mdata["base_power_kw"],
            efficiency    = mdata["efficiency"],
        )
        for mid, mdata in sorted(machines.items())
        if mdata["type"] == machine_type
    ]

    if not result:
        raise HTTPException(status_code=404,
                            detail=f"Tidak ada mesin bertipe '{machine_type}'")
    return result
