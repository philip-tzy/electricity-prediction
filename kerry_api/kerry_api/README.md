# Kerry AI Production Scheduling — Backend API

FastAPI backend untuk sistem penjadwalan produksi berbasis XGBoost.

## Struktur Project

```
kerry_api/
├── main.py                  ← Entry point FastAPI
├── requirements.txt         ← Dependencies
├── model_files/             ← Taruh 3 file output training di sini
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── bom_routing.json
├── core/
│   ├── model_loader.py      ← Singleton loader model ke memori
│   └── predictor.py         ← Engine: BOM ekspansi, prediksi, optimizer
├── routers/
│   ├── products.py          ← GET /api/products
│   ├── machines.py          ← GET /api/machines
│   ├── schedule.py          ← POST /api/predict-schedule
│   └── kpi.py               ← GET /api/kpi/summary
└── schemas/
    └── schedule.py          ← Pydantic request & response schemas
```

## Setup & Jalankan

### 1. Siapkan file model
```bash
mkdir model_files
# Copy 3 file dari hasil training Colab:
cp /path/to/model.pkl        model_files/
cp /path/to/preprocessor.pkl model_files/
cp /path/to/bom_routing.json model_files/
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan server
Jalankan dari salah satu folder berikut (ada launcher `main.py`):
- root project `kerry/`
- `kerry/kerry_api/`
- `kerry/kerry_api/kerry_api/` (package utama)

```bash
# Development (dengan auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Buka dokumentasi interaktif
- Swagger UI : http://localhost:8000/docs
- ReDoc      : http://localhost:8000/redoc

---

## Endpoint Ringkasan

| Method | Endpoint                     | Fungsi                                      |
|--------|------------------------------|---------------------------------------------|
| GET    | `/`                          | Health check                                |
| GET    | `/api/products`              | Daftar semua produk + rute BOM              |
| GET    | `/api/products/{id}`         | Detail satu produk                          |
| GET    | `/api/machines`              | Daftar semua mesin                          |
| GET    | `/api/machines/{id}`         | Detail satu mesin                           |
| GET    | `/api/machines/type/{type}`  | Mesin berdasarkan tipe                      |
| POST   | `/api/predict-schedule`      | **CORE** — Generate jadwal optimal dari order |
| GET    | `/api/schedule`              | Daftar semua jadwal dalam cache             |
| GET    | `/api/schedule/{order_id}`   | Ambil jadwal dari cache                     |
| DELETE | `/api/schedule/{order_id}`   | Hapus jadwal dari cache                     |
| GET    | `/api/kpi/summary`           | Agregasi KPI semua jadwal                   |
| GET    | `/api/kpi/{order_id}`        | KPI detail + breakdown per mesin & produk   |

---

## Contoh Request & Response

### POST /api/predict-schedule

**Request:**
```json
{
  "month": "2025-11",
  "orders": [
    {"product_id": "P001", "quantity_mt": 20},
    {"product_id": "P004", "quantity_mt": 100},
    {"product_id": "P006", "quantity_mt": 150}
  ]
}
```

**Response (ringkasan):**
```json
{
  "month": "2025-11",
  "generated_at": "2025-11-01T08:00:00",
  "kpi": {
    "total_kwh": 142380,
    "total_mt": 270,
    "kwh_per_mt": 527.33,
    "operating_days": 14,
    "saving_kwh": 25628,
    "saving_pct": 18.0,
    "days_saved": 8,
    "prd_kpi_target_met": true
  },
  "schedule": [
    {
      "machine_id": "M001",
      "machine_type": "Mixer",
      "product_label": "Produk Standard",
      "shift": "Shift 3",
      "start_date": "2025-11-03",
      "end_date": "2025-11-03",
      "predicted_kwh": 481.5,
      "instruction": "Jalankan M001 (Mixer) untuk Produk Standard sebanyak 20 MT pada Senin, 03 Nov 2025, Shift 3.",
      "wip_note": null
    }
  ],
  "machines_standby": ["M003", "M006", "M007"]
}
```

---

## Integrasi Frontend

Dashboard HTML ada di `../../frontend/kerry_dashboard.html` (folder `frontend/` di root project).

Setelah server jalan, buka:

| URL | Fungsi |
|-----|--------|
| http://localhost:8000/dashboard | Dashboard UI (terintegrasi API) |
| http://localhost:8000/docs | Swagger API |
| http://localhost:8000/api/health | Health check |

Frontend memanggil endpoint yang sama (same-origin):

```javascript
const response = await fetch('/api/predict-schedule', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    month: '2025-11',
    orders: [
      { product_id: 'P004', quantity_mt: 100 },
      { product_id: 'P006', quantity_mt: 150 },
    ]
  })
});

const data = await response.json();
console.log(data.order_id);           // ID cache jadwal
console.log(data.kpi);                // KPI panel
console.log(data.schedule);           // Instruksi operator
console.log(data.machines_standby);   // Mesin tidak dipakai
```

> Tip: set `DEMO_MODE = true` di `kerry_dashboard.html` jika ingin uji UI tanpa model/API.
