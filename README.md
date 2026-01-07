# API สำหรับการประเมินราคาและค่าสูญเสียของรถยนต์

บริการ Machine Learning แบบ FastAPI สำหรับการทำนายราคารถยนต์และการประมาณค่าสูญเสียของยานพาหนะในตลาดไทย ระบบให้แถบราคา (เขียว/เหลือง/แดง) เพื่อแนะนำกลยุทธ์การขายและสร้างกราฟราคาแบบภาพ

## คุณสมบัติ

- **การทำนายราคา**: ใช้โมเดล LightGBM quantile regression (q20, q50, q80) เพื่อทำนายช่วงราคา
- **แถบราคา**: ส่งคืนกลยุทธ์การกำหนดราคาสามแบบ:
  - **เขียว (ขายเร็ว)**: ต่ำกว่าราคากลางตลาด 8-12%
  - **เหลือง (ราคากลาง)**: ราคากลางตลาด
  - **แดง (รอขาย)**: สูงกว่าราคากลางตลาด 10-18%
- **การประมาณค่าสูญเสีย**: คาดการณ์มูลค่ารถยนต์ในอนาคตโดยใช้ robust log-linear regression
- **กราฟภาพ**: สร้างกราฟแท่ง PNG แสดงแถบราคา
- **การเชื่อมต่อ MySQL**: เชื่อมต่อกับฐานข้อมูลสำหรับข้อมูลราคาประวัติศาสตร์

## โครงสร้างโปรเจกต์

```
.
├── entry.py                 # แอปพลิเคชัน FastAPI พร้อม endpoints ทั้งหมด
├── price_helper.py          # ตรรกะการทำนายราคาและการโหลดโมเดล
├── depreciation.py          # โมดูลการประมาณค่าสูญเสีย
├── models/
│   └── price_quantiles_v3/  # โมเดล LightGBM ที่ฝึกแล้ว (q20, q50, q80)
│       ├── q20_lgbm.pkl
│       ├── q50_lgbm.pkl
│       ├── q80_lgbm.pkl
│       ├── feature_config.json
│       ├── group_medians.csv
│       └── metrics.json
├── artifacts/              # artifacts ของโมเดลค่าสูญเสีย
├── data/                   # ข้อมูลรายการรถยนต์ที่ประมวลผลแล้ว
├── requirements.txt        # dependencies หลัก
└── requirements_api.txt    # dependencies เฉพาะ API
```

## การติดตั้ง

### ความต้องการเบื้องต้น

- Python 3.12+
- ฐานข้อมูล MySQL (สำหรับฟีเจอร์ค่าสูญเสีย)
- Virtual environment (แนะนำ)

### การตั้งค่า

1. Clone repository และนำทางไปยังไดเรกทอรีโปรเจกต์:
```bash
cd "Ai program test"
```

2. สร้างและเปิดใช้งาน virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
```

4. กำหนดค่าตัวแปรสภาพแวดล้อม (สร้างไฟล์ `.env` สำหรับฟีเจอร์ค่าสูญเสีย):
```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=carpricing
MYSQL_USER=car_user
MYSQL_PASSWORD=StrongPassword123
```

5. ตรวจสอบให้แน่ใจว่าไฟล์โมเดลมีอยู่ใน `models/price_quantiles_v3/`:
   - `q20_lgbm.pkl`
   - `q50_lgbm.pkl`
   - `q80_lgbm.pkl`
   - `feature_config.json`
   - `group_medians.csv` (ไม่บังคับ)

## การใช้งาน

### การเริ่มเซิร์ฟเวอร์

```bash
python entry.py
```

API จะพร้อมใช้งานที่ `http://0.0.0.0:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
ส่งคืนสถานะสุขภาพของเซิร์ฟเวอร์

**Response:**
```json
{
  "status": "healthy"
}
```

#### 2. Price Model Health
```http
GET /health/price_model
```
ตรวจสอบว่าโมเดลการทำนายราคาถูกโหลดและพร้อมใช้งานหรือไม่

**Response:**
```json
{
  "ready": true,
  "path": "models/price_quantiles_v3"
}
```

#### 3. Price Prediction
```http
POST /price
Content-Type: application/json
```

**Request Body:**
```json
{
  "make": "TOYOTA",
  "model": "Yaris",
  "year": 2019,
  "mileage_km_num": 30000,
  "submodel": "1.5 E" (ไม่บังคับ),
  "gear": "AT" (ไม่บังคับ),
  "color": "White" (ไม่บังคับ)
}
```

**Response:**
```json
{
  "green_low": 450000,
  "green_median": 470000,
  "green_high": 480000,
  "yellow": 520000,
  "red_low": 570000,
  "red_median": 590000,
  "red_high": 610000,
  "confidence": 0.0
}
```

#### 4. Price Graph
```http
POST /price_graph
Content-Type: application/json
```

**Request Body:** เหมือนกับ endpoint `/price`

**Response:** ภาพ PNG (image/png) แสดงกราฟแท่งพร้อมแถบราคาสีเขียว เหลือง และแดง

#### 5. Depreciation Estimation
```http
POST /depreciation
Content-Type: application/json
```

**Request Body:**
```json
{
  "make": "TOYOTA",
  "model": "Yaris",
  "year": 2019,
  "mileage_km_num": 30000,
  "submodel": "1.5 E" (ไม่บังคับ),
  "horizon_years": 5 (ไม่บังคับ, ค่าเริ่มต้น: 5)
}
```

**Response:**
```json
{
  "brand": "TOYOTA",
  "model": "Yaris",
  "submodel": "1.5 E",
  "year": 2019,
  "mileage": 30000,
  "sample_size": 150,
  "predicted_price_now": 520000.00,
  "lower_now": 375000.00,
  "upper_now": 720000.00,
  "annual_projection": [
    {
      "calendar_year": 2025,
      "age": 6,
      "mileage": 42000,
      "price": 480000.00,
      "lower": 345000.00,
      "upper": 670000.00,
      "depreciation_from_now_pct": 7.69
    }
    // ... ปีอื่นๆ เพิ่มเติม
  ],
  "km_per_year_assumed": 12000,
  "notes": "Robust log-linear model: log(price) ~ age + log(mileage+1)..."
}
```

#### 6. Reload Price Model
```http
POST /admin/reload_price_model
```
โหลดโมเดลการทำนายราคาใหม่จากดิสก์

**Response:**
```json
{
  "ok": true,
  "path": "models/price_quantiles_v3"
}
```

#### 7. Test Depreciation
```http
GET /test_dep
```
ทดสอบว่า endpoint ค่าสูญเสียพร้อมใช้งานหรือไม่

## รายละเอียดโมเดล

### โมเดลการทำนายราคา

- **อัลกอริทึม**: LightGBM (Gradient Boosting)
- **แนวทาง**: Quantile Regression (q20, q50, q80)
- **เป้าหมาย**: log(price) ในหน่วยบาทไทย
- **Features**:
  - Categorical: brand, model, submodel, gear, color
  - Numeric: year, age, mileage, mileage_per_year, log_mileage, sqrt_mileage, age_x_mileage, mileage_per_age
- **Post-processing**:
  - ผสมการทำนายกับ group medians (น้ำหนัก 30%)
  - ใช้ sanity caps (q20 ≥ 65% ของ q50, q80 ≤ 175% ของ q50)
  - ปัดเศษราคา: 1,000 บาท สำหรับ < 1M, 5,000 บาท สำหรับ ≥ 1M

### โมเดลค่าสูญเสีย

- **อัลกอริทึม**: HuberRegressor (robust linear regression)
- **โมเดล**: `log(price) ~ age + log(mileage+1)`
- **แหล่งข้อมูล**: ฐานข้อมูล MySQL (ตาราง `car_listings_master`)
- **Features**:
  - Age (current_year - model_year)
  - Log-transformed mileage
- **Output**: การประมาณราคาปัจจุบันพร้อมแถบความไม่แน่นอนและการคาดการณ์อนาคต

## กลยุทธ์แถบราคา

ระบบคำนวณแถบราคาสามแบบเทียบกับราคากลางตลาด (q50):

- **แถบเขียว** (ขายเร็ว): ต่ำกว่ากลาง 8-12%
  - `green_low`: ต่ำกว่ากลาง 12%
  - `green_median`: ต่ำกว่ากลาง 10%
  - `green_high`: ต่ำกว่ากลาง 8%

- **แถบเหลือง** (ราคากลางตลาด): เท่ากับการทำนาย q50

- **แถบแดง** (รอขาย): สูงกว่ากลาง 10-18%
  - `red_low`: สูงกว่ากลาง 10%
  - `red_median`: สูงกว่ากลาง 14%
  - `red_high`: สูงกว่ากลาง 18%

## Dependencies

### Core Dependencies
- `pandas==2.2.2`
- `numpy==1.26.4`
- `lightgbm==4.5.0`
- `scikit-learn==1.5.2`
- `joblib==1.4.2`
- `python-dotenv==1.0.1`
- `pyarrow==17.0.0`

### API Dependencies
- `fastapi[standard]`
- `uvicorn[standard]`
- `SQLAlchemy`
- `PyMySQL`
- `matplotlib` (สำหรับการสร้างกราฟ)

## การพัฒนา

### การฝึกโมเดล

โมเดลราคาถูกฝึกโดยใช้สคริปต์ใน repository:
- `train_model.py` / `train_model_v2.py` / `train_model_v3.py`

### การประมวลผลข้อมูล

สคริปต์การทำความสะอาดและประมวลผลข้อมูลต่างๆ:
- `clean_data.py` / `clean_data_v3.py`
- Jupyter notebooks สำหรับ EDA และ feature engineering

## การจัดการข้อผิดพลาด

- **503 Service Unavailable**: โมเดลราคาไม่ได้ถูกโหลด
- **400 Bad Request**: พารามิเตอร์อินพุตไม่ถูกต้อง
- **500 Internal Server Error**: ข้อผิดพลาดจากการทำนายโมเดลหรือฐานข้อมูล

## หมายเหตุ

- โมเดลราคาต้องการโมเดล quantile ทั้งสาม (q20, q50, q80) ที่ถูกโหลด
- การประมาณค่าสูญเสียต้องการการเชื่อมต่อฐานข้อมูล MySQL
- ราคาทั้งหมดถูกส่งคืนในหน่วยบาทไทย (THB)
- ระบบทำให้อินพุต categorical (brand, model) เป็นตัวพิมพ์ใหญ่
- ฟิลด์ที่ไม่บังคับที่หายไปจะถูกจัดการอย่างเหมาะสมด้วยค่าเริ่มต้น

## สิทธิ์การใช้งาน

[ระบุสิทธิ์การใช้งานของคุณที่นี่]

## ผู้เขียน

[ชื่อ/องค์กรของคุณ]

---

# Car Pricing & Depreciation API

A FastAPI-based machine learning service for predicting car prices and estimating depreciation for vehicles in the Thai market. The system provides price bands (green/yellow/red) to guide selling strategies and generates visual price charts.

## Features

- **Price Prediction**: Uses LightGBM quantile regression models (q20, q50, q80) to predict price ranges
- **Price Bands**: Returns three pricing strategies:
  - **Green (sell fast)**: 8-12% below market median
  - **Yellow (median)**: Market median price
  - **Red (hold out)**: 10-18% above market median
- **Depreciation Estimation**: Projects future vehicle values using robust log-linear regression
- **Visual Charts**: Generates PNG bar charts showing price bands
- **MySQL Integration**: Connects to database for historical pricing data

## Project Structure

```
.
├── entry.py                 # FastAPI application with all endpoints
├── price_helper.py          # Price prediction logic and model loading
├── depreciation.py          # Depreciation estimation module
├── models/
│   └── price_quantiles_v3/  # Trained LightGBM models (q20, q50, q80)
│       ├── q20_lgbm.pkl
│       ├── q50_lgbm.pkl
│       ├── q80_lgbm.pkl
│       ├── feature_config.json
│       ├── group_medians.csv
│       └── metrics.json
├── artifacts/              # Depreciation model artifacts
├── data/                   # Processed car listing data
├── requirements.txt        # Core dependencies
└── requirements_api.txt    # API-specific dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- MySQL database (for depreciation features)
- Virtual environment (recommended)

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd "Ai program test"
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
```

4. Configure environment variables (create a `.env` file for depreciation features):
```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=carpricing
MYSQL_USER=car_user
MYSQL_PASSWORD=StrongPassword123
```

5. Ensure model files are present in `models/price_quantiles_v3/`:
   - `q20_lgbm.pkl`
   - `q50_lgbm.pkl`
   - `q80_lgbm.pkl`
   - `feature_config.json`
   - `group_medians.csv` (optional)

## Usage

### Starting the Server

```bash
python entry.py
```

The API will be available at `http://0.0.0.0:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
Returns server health status.

**Response:**
```json
{
  "status": "healthy"
}
```

#### 2. Price Model Health
```http
GET /health/price_model
```
Checks if the price prediction model is loaded and ready.

**Response:**
```json
{
  "ready": true,
  "path": "models/price_quantiles_v3"
}
```

#### 3. Price Prediction
```http
POST /price
Content-Type: application/json
```

**Request Body:**
```json
{
  "make": "TOYOTA",
  "model": "Yaris",
  "year": 2019,
  "mileage_km_num": 30000,
  "submodel": "1.5 E" (optional),
  "gear": "AT" (optional),
  "color": "White" (optional)
}
```

**Response:**
```json
{
  "green_low": 450000,
  "green_median": 470000,
  "green_high": 480000,
  "yellow": 520000,
  "red_low": 570000,
  "red_median": 590000,
  "red_high": 610000,
  "confidence": 0.0
}
```

#### 4. Price Graph
```http
POST /price_graph
Content-Type: application/json
```

**Request Body:** Same as `/price` endpoint

**Response:** PNG image (image/png) showing a bar chart with green, yellow, and red price bands

#### 5. Depreciation Estimation
```http
POST /depreciation
Content-Type: application/json
```

**Request Body:**
```json
{
  "make": "TOYOTA",
  "model": "Yaris",
  "year": 2019,
  "mileage_km_num": 30000,
  "submodel": "1.5 E" (optional),
  "horizon_years": 5 (optional, default: 5)
}
```

**Response:**
```json
{
  "brand": "TOYOTA",
  "model": "Yaris",
  "submodel": "1.5 E",
  "year": 2019,
  "mileage": 30000,
  "sample_size": 150,
  "predicted_price_now": 520000.00,
  "lower_now": 375000.00,
  "upper_now": 720000.00,
  "annual_projection": [
    {
      "calendar_year": 2025,
      "age": 6,
      "mileage": 42000,
      "price": 480000.00,
      "lower": 345000.00,
      "upper": 670000.00,
      "depreciation_from_now_pct": 7.69
    }
    // ... more years
  ],
  "km_per_year_assumed": 12000,
  "notes": "Robust log-linear model: log(price) ~ age + log(mileage+1)..."
}
```

#### 6. Reload Price Model
```http
POST /admin/reload_price_model
```
Reloads the price prediction models from disk.

**Response:**
```json
{
  "ok": true,
  "path": "models/price_quantiles_v3"
}
```

#### 7. Test Depreciation
```http
GET /test_dep
```
Tests if the depreciation endpoint is ready.

## Model Details

### Price Prediction Model

- **Algorithm**: LightGBM (Gradient Boosting)
- **Approach**: Quantile Regression (q20, q50, q80)
- **Target**: log(price) in THB
- **Features**:
  - Categorical: brand, model, submodel, gear, color
  - Numeric: year, age, mileage, mileage_per_year, log_mileage, sqrt_mileage, age_x_mileage, mileage_per_age
- **Post-processing**:
  - Blends predictions with group medians (30% weight)
  - Applies sanity caps (q20 ≥ 65% of q50, q80 ≤ 175% of q50)
  - Rounds prices: 1,000 THB for < 1M, 5,000 THB for ≥ 1M

### Depreciation Model

- **Algorithm**: HuberRegressor (robust linear regression)
- **Model**: `log(price) ~ age + log(mileage+1)`
- **Data Source**: MySQL database (`car_listings_master` table)
- **Features**:
  - Age (current_year - model_year)
  - Log-transformed mileage
- **Output**: Current price estimate with uncertainty bands and future projections

## Price Band Strategy

The system calculates three price bands relative to the market median (q50):

- **Green Band** (Sell Fast): 8-12% below median
  - `green_low`: 12% below median
  - `green_median`: 10% below median
  - `green_high`: 8% below median

- **Yellow Band** (Market Median): Equal to q50 prediction

- **Red Band** (Hold Out): 10-18% above median
  - `red_low`: 10% above median
  - `red_median`: 14% above median
  - `red_high`: 18% above median

## Dependencies

### Core Dependencies
- `pandas==2.2.2`
- `numpy==1.26.4`
- `lightgbm==4.5.0`
- `scikit-learn==1.5.2`
- `joblib==1.4.2`
- `python-dotenv==1.0.1`
- `pyarrow==17.0.0`

### API Dependencies
- `fastapi[standard]`
- `uvicorn[standard]`
- `SQLAlchemy`
- `PyMySQL`
- `matplotlib` (for chart generation)

## Development

### Model Training

The price models are trained using scripts in the repository:
- `train_model.py` / `train_model_v2.py` / `train_model_v3.py`

### Data Processing

Various data cleaning and processing scripts:
- `clean_data.py` / `clean_data_v3.py`
- Jupyter notebooks for EDA and feature engineering

## Error Handling

- **503 Service Unavailable**: Price model not loaded
- **400 Bad Request**: Invalid input parameters
- **500 Internal Server Error**: Model prediction or database errors

## Notes

- The price model requires all three quantile models (q20, q50, q80) to be loaded
- Depreciation estimation requires a MySQL database connection
- All prices are returned in Thai Baht (THB)
- The system normalizes categorical inputs (brand, model) to uppercase
- Missing optional fields are handled gracefully with defaults

## License

[Specify your license here]

## Author

[Your name/organization]
