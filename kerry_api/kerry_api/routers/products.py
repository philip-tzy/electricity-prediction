"""
Router: Products & BOM
GET /api/products       → Daftar semua produk + rute BOM lengkap
GET /api/products/{id}  → Detail satu produk
"""

from fastapi import APIRouter, HTTPException
from typing import List

from core.model_loader import ModelLoader
from schemas.schedule import ProductInfo, RouteStep

router = APIRouter()


@router.get("/products", response_model=List[ProductInfo], summary="Daftar semua produk + rute BOM")
def get_products():
    """
    Mengembalikan semua produk yang terdaftar di BOM beserta rute mesinnya.

    Dipakai frontend untuk:
    - Mengisi dropdown pilihan produk di form input order
    - Menampilkan preview rute mesin sebelum order dikirim
    """
    _, _, bom_routing = ModelLoader.get()
    products = bom_routing["products"]

    result = []
    for pid, pdata in products.items():
        result.append(ProductInfo(
            product_id    = pid,
            label         = pdata["label"],
            energy_factor = pdata["energy_factor"],
            n_steps       = len(pdata["route"]),
            route         = [
                RouteStep(
                    step         = s["step"],
                    machine_type = s["machine_type"],
                    hours_per_mt = s["hours_per_mt"],
                    desc         = s["desc"],
                )
                for s in pdata["route"]
            ],
        ))

    return sorted(result, key=lambda x: x.product_id)


@router.get("/products/{product_id}", response_model=ProductInfo, summary="Detail satu produk")
def get_product(product_id: str):
    """
    Mengembalikan detail satu produk berdasarkan product_id (P001–P007).
    """
    _, _, bom_routing = ModelLoader.get()
    products = bom_routing["products"]

    if product_id not in products:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' tidak ditemukan. "
                   f"Tersedia: {sorted(products.keys())}"
        )

    pdata = products[product_id]
    return ProductInfo(
        product_id    = product_id,
        label         = pdata["label"],
        energy_factor = pdata["energy_factor"],
        n_steps       = len(pdata["route"]),
        route         = [
            RouteStep(
                step         = s["step"],
                machine_type = s["machine_type"],
                hours_per_mt = s["hours_per_mt"],
                desc         = s["desc"],
            )
            for s in pdata["route"]
        ],
    )
