# ระบบคาดการณ์ราคารถยนต์และการเสื่อมมูลค่า (ค่าเสื่อมราคา)

ระบบ Machine Learning ที่พัฒนาด้วย FastAPI สำหรับคาดการณ์ราคารถยนต์และการเสื่อมมูลค่า (ค่าเสื่อมราคา) ของรถยนต์ในตลาดประเทศไทย  

ระบบจะแสดงช่วงราคาแบบ Green / Yellow / Red เพื่อช่วยในการตัดสินใจตั้งราคาขาย และสามารถสร้างกราฟแสดงช่วงราคาในรูปแบบภาพได้

## คุณสมบัติ (Features)

- **การคาดการณ์ราคา**: ใช้โมเดล LightGBM แบบ Quantile Regression (q20, q50, q80) เพื่อคาดการณ์ช่วงราคารถยนต์
- **ช่วงราคา (Price Bands)**: แสดงกลยุทธ์การตั้งราคา 3 รูปแบบ
  - **Green (ขายเร็ว)**: ต่ำกว่าราคากลางตลาดประมาณ 8–12%
  - **Yellow (ราคากลาง)**: เท่ากับราคากลางของตลาด
  - **Red (รอขาย)**: สูงกว่าราคากลางตลาดประมาณ 10–18%
- **การประเมินการเสื่อมมูลค่า (ค่าเสื่อมราคา)**: คาดการณ์มูลค่ารถยนต์ในอนาคตด้วยโมเดลถดถอยแบบ log-linear ที่มีความทนทานต่อ outlier
- **กราฟแสดงผลราคา**: สร้างกราฟแท่งในรูปแบบไฟล์ PNG เพื่อแสดงช่วงราคา
- **เชื่อมต่อ MySQL**: ใช้ข้อมูลราคาย้อนหลังจากฐานข้อมูลเพื่อการคำนวณ
---

## โครงสร้างโปรเจกต์ (Project Structure)


```
.
├── entry.py                 # แอปพลิเคชัน FastAPI หลัก รวมทุก API endpoint
├── price_helper.py          # Logic การคาดการณ์ราคาและการโหลดโมเดล
├── depreciation.py          # โมดูลสำหรับคำนวณการเสื่อมมูลค่า
├── models/
│   └── price_quantiles_v3/  # โมเดล LightGBM ที่ฝึกแล้ว (q20, q50, q80)
│       ├── q20_lgbm.pkl
│       ├── q50_lgbm.pkl
│       ├── q80_lgbm.pkl
│       ├── feature_config.json
│       ├── group_medians.csv
│       └── metrics.json
├── artifacts/              # ไฟล์ผลลัพธ์และอาร์ติแฟกต์ของโมเดลการเสื่อมมูลค่า
├── data/                   # ข้อมูลรถที่ผ่านการประมวลผลแล้ว
├── requirements.txt        # dependencies หลักของโปรเจกต์
└── requirements_api.txt    # dependencies สำหรับ API โดยเฉพาะ
```
---


## การติดตั้ง (Installation)

### ข้อกำหนดเบื้องต้น (Prerequisites)

- Docker (แนะนำ)
- หรือ Python 3.12+ (สำหรับการรันแบบ Local)
- ฐานข้อมูล MySQL (จำเป็นสำหรับฟีเจอร์การเสื่อมมูลค่า)
- Virtual environment (แนะนำ)

### โคลนรีโพซิทอรี (จำเป็น)
```bash
git clone https://github.com/kaywabunny/CarAI.git
cd CarAI
git checkout master
```

---

## การติดตั้งด้วย Docker (แนะนำ)
✅ วิธีนี้เป็นวิธีที่แนะนำและง่ายที่สุดในการรันโปรเจกต์
Docker จะช่วยจัดการ dependencies, แยก environment และควบคุมความสม่ำเสมอของ runtime ให้เรียบร้อย

#### 1. เปิดใช้งาน Docker

ตรวจสอบให้แน่ใจว่า Docker Desktop (หรือ Docker Engine) กำลังทำงานอยู่

#### 2. สร้างและเริ่มต้นบริการ

```bash
docker compose up --build
```

#### 3. โหลดโมเดลคาดการณ์ราคาใหม่

เมื่อ container เริ่มทำงานแล้ว ให้รันคำสั่ง:
```bash
curl -X POST http://localhost:8000/admin/reload_price_model

```

#### 4. เข้าใช้งาน API
```bash
http://localhost:8000

```
---


## API Endpoints

### 1. ตรวจสอบสถานะเซิร์ฟเวอร์ (Health Check)
```http
GET /health
```
ใช้สำหรับตรวจสอบว่า API ทำงานอยู่หรือไม่

**ตัวอย่าง**
```bash
POST http://localhost:8000/health

```

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 2. ตรวจสอบสถานะโมเดลราคา
```http
POST /health/price_model
```
ตรวจสอบว่าโมเดลคาดการณ์ราคาถูกโหลดและพร้อมใช้งานหรือไม่

**ตัวอย่าง**
```bash
POST http://localhost:8000/health/price_model

```

**Response:**
```json
{
  "ready": true,
  "path": "models/price_quantiles_v3"
}
```
---

### 3. คาดการณ์ราคา (POST – JSON Body)
```http
POST /price
Content-Type: application/json
```

ใช้สำหรับคาดการณ์ช่วงราคารถยนต์ (Green / Yellow / Red)

**ตัวอย่าง**
```bash
POST http://localhost:8000/price

```

**Request Body:**
```json
{
  "make": "Toyota",
  "model": "Yaris",
  "year": 2012,
  "mileage_km_num": 30000

}

```

**Response:**
```json
{
    "green_low": 197000,
    "green_median": 202000,
    "green_high": 206000,
    "yellow": 224000,
    "red_low": 247000,
    "red_median": 256000,
    "red_high": 265000,
    "estimate_basis": "based_on_comparable_listings",
    "confidence": 0.6,
    "sample_size": 8,
    "bandwidth_clamped": false
}
```
---

### 4. กราฟแสดงช่วงราคา
```http
POST /price_graph
Content-Type: application/json
```
ใช้สำหรับสร้างกราฟแสดงช่วงราคา Green / Yellow / Red

**Request Body:** เหมือนกับ `/price` endpoint

**Response:** ไฟล์ภาพ PNG (image/png) แสดงกราฟแท่งของช่วงราคา

---

### 5. ประเมินการเสื่อมมูลค่า (ค่าเสื่อมราคา)
```http
POST /depreciation
Content-Type: application/json
```

ใช้สำหรับคาดการณ์มูลค่ารถยนต์ในปัจจุบันและอนาคต

**Request Body:**
```json
{"make": "Toyota","model": "Yaris","year": 2019,"horizon_years": 6,"mileage_km_num": 30000.0}
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
---

### 6. ทดสอบระบบการเสื่อมมูลค่า
```http
GET /test_dep
```
ใช้สำหรับตรวจสอบว่า endpoint การประเมินการเสื่อมมูลค่าพร้อมใช้งานหรือไม่

---


## การติดตั้งแบบ Manual (Local Environment – ตัวเลือกเสริม)
  ⚠️ วิธีนี้เหมาะสำหรับการพัฒนา การดีบัก หรือในกรณีที่ไม่สามารถใช้ Docker ได้


#### 1. ไปยังไดเรกทอรีของโปรเจกต์
หลังจากโคลนรีโพซิทอรีแล้ว:
```bash
cd CarAI
```

#### 2. สร้างและเปิดใช้งาน Virtual Environment
หลังจากโคลนรีโพซิทอรีแล้ว:
```bash
python -m venv venv
```
**Windows**
```bash
venv\Scripts\activate
```
**Linux / macOS**
```bash
source venv/bin/activate
```

#### 3. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_api.txt

```

#### 4. ตั้งค่า Environment Variables
สร้างไฟล์ .env ในโฟลเดอร์หลักของโปรเจกต์ (จำเป็นสำหรับฟีเจอร์การเสื่อมมูลค่า):

```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=carpricing
MYSQL_USER=car_user
MYSQL_PASSWORD=StrongPassword123
```

#### 5. ตรวจสอบว่าไฟล์โมเดลอยู่ครบใน `models/price_quantiles_v3/`:
   - `q20_lgbm.pkl`
   - `q50_lgbm.pkl`
   - `q80_lgbm.pkl`
   - `feature_config.json`
   - `group_medians.csv` (ไม่บังคับ)

#### 6. เริ่มต้นรันเซิร์ฟเวอร์
จากโฟลเดอร์หลักของโปรเจกต์:
```bash
python entry.py
```
API จะพร้อมใช้งานที่:
```bash
http://localhost:8000
```

#### 7. โหลดโมเดลคาดการณ์ราคาใหม่ (จำเป็น)
เปิด terminal ใหม่แล้วรันคำสั่ง:
```bash
curl -X POST http://localhost:8000/admin/reload_price_model
```

#### 8. ทดสอบการทำงานของ API
สามารถทดสอบ endpoint ต่าง ๆ ได้ เช่น:
   - `POST /price`
   - `POST /price_graph`
   - `POST /depreciation`
   - `GET /health`
**ตัวอย่าง:**
```bash
http://localhost:8000/price
```
---

## รายละเอียดของโมเดล (Model Details)

### โมเดลคาดการณ์ราคา

- **อัลกอริทึม**: LightGBM (Gradient Boosting)
- **แนวทางการฝึกโมเดล**: Quantile Regression (q20, q50, q80)
- **ตัวแปรเป้าหมาย (Target)**: log(price) หน่วยเป็นบาทไทย (THB)
- **Features**:
  - ตัวแปรเชิงหมวดหมู่ (Categorical): brand, model, submodel, gear, color
  - ตัวแปรเชิงตัวเลข (Numeric): year, age, mileage, mileage_per_year, log_mileage, sqrt_mileage, age_x_mileage, mileage_per_age
- **การประมวลผลหลังการทำนาย (Post-processing)**:
  - ผสมผลลัพธ์จากโมเดลกับค่ามัธยฐานของกลุ่มข้อมูล (group median) โดยให้น้ำหนัก 30%
  - ใช้ sanity caps เพื่อจำกัดค่าผลลัพธ์  
    (q20 ≥ 65% ของ q50 และ q80 ≤ 175% ของ q50)
  - ปัดเศษราคา:  
    - 1,000 บาท สำหรับราคาต่ำกว่า 1 ล้านบาท  
    - 5,000 บาท สำหรับราคาตั้งแต่ 1 ล้านบาทขึ้นไป

### โมเดลการเสื่อมมูลค่า (ค่าเสื่อมราคา)

- **อัลกอริทึม**: HuberRegressor (robust linear regression)
- **สมการของโมเดล**: `log(price) ~ age + log(mileage+1)`
- **แหล่งข้อมูล**: ฐานข้อมูล MySQL (ตาราง `car_listings_master`)
- **Features**:
  - อายุรถ (age = current_year - model_year)
  - ระยะทางสะสมแบบ log-transformed (log(mileage))
- **ผลลัพธ์**:
  - การประเมินราคารถในปัจจุบัน
  - ช่วงความไม่แน่นอนของราคา (uncertainty bands)
  - การคาดการณ์ราคาในอนาคต

## กลยุทธ์ช่วงราคา (Price Band Strategy)

ระบบจะคำนวณช่วงราคา 3 ระดับ โดยอ้างอิงจากราคากลางของตลาด (q50):

- **Green Band (ขายเร็ว)**: ต่ำกว่าราคากลาง 8–12%
  - `green_low`: ต่ำกว่าราคากลาง 12%
  - `green_median`: ต่ำกว่าราคากลาง 10%
  - `green_high`: ต่ำกว่าราคากลาง 8%

- **Yellow Band (ราคากลางตลาด)**: เท่ากับค่าที่โมเดล q50 คาดการณ์

- **Red Band (รอขาย / เน้นกำไร)**: สูงกว่าราคากลาง 10–18%
  - `red_low`: สูงกว่าราคากลาง 10%
  - `red_median`: สูงกว่าราคากลาง 14%
  - `red_high`: สูงกว่าราคากลาง 18%

---

## Dependencies

### Core Dependencies
- `pandas==2.2.2`
- `numpy==1.26.4`
- `lightgbm==4.5.0`
- `scikit-learn==1.5.2`
- `joblib==1.4.2`
- `python-dotenv==1.0.1`
- `pyarrow==17.0.0`
- `pydantic==2.9.0`
- `mysql-connector-python==9.0.0`
- `cryptography`
- `tqdm==4.66.5`

### API Dependencies
- `fastapi[standard]`
- `uvicorn[standard]`
- `SQLAlchemy`
- `PyMySQL`
- `matplotlib` (ใช้สำหรับสร้างกราฟ)

---

## การพัฒนา (Development)

### การฝึกโมเดล

โมเดลคาดการณ์ราคาถูกฝึกด้วยสคริปต์:
- `train_model_v3.py`

### การประมวลผลข้อมูล

สคริปต์สำหรับทำความสะอาดและเตรียมข้อมูล:
- `clean_data_v3.py`



## การจัดการข้อผิดพลาด (Error Handling)

- **503 Service Unavailable**: โมเดลคาดการณ์ราคายังไม่ถูกโหลด
- **400 Bad Request**: พารามิเตอร์อินพุตไม่ถูกต้อง
- **500 Internal Server Error**: เกิดข้อผิดพลาดระหว่างการคาดการณ์หรือการเชื่อมต่อฐานข้อมูล  
  (แนะนำให้ลอง reload โมเดลอีกครั้ง)

---

## หมายเหตุ (Notes)

- โมเดลคาดการณ์ราคาจำเป็นต้องโหลดครบทั้ง 3 quantile models (q20, q50, q80)
- การประเมินการเสื่อมมูลค่าจำเป็นต้องเชื่อมต่อฐานข้อมูล MySQL
- ราคาทั้งหมดแสดงผลเป็นสกุลเงินบาท (THB)
- ระบบจะปรับค่าตัวแปรเชิงหมวดหมู่ (brand, model) เป็นตัวพิมพ์ใหญ่โดยอัตโนมัติ
- ฟิลด์ที่เป็น optional หากไม่ระบุ ระบบจะจัดการด้วยค่าเริ่มต้นอย่างเหมาะสม

---

## ใบอนุญาต (License)

-

## ผู้จัดทำ (Author)

-


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

---

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

---

## Installation

### Prerequisites

- Docker (recommended)
- OR Python 3.12+ (for local setup)
- MySQL database (for depreciation features)
- Virtual environment (recommended)

### Clone the Repository (Required)
```bash
git clone https://github.com/kaywabunny/CarAI.git
cd CarAI
git checkout master
```

---
## Docker Setup (Recommended)
✅ This is the recommended and easiest way to run the project.
Docker handles dependencies, environment isolation, and runtime consistency. 

### 1. Start Docker

Ensure Docker Desktop (or Docker Engine) is running.

### 2. Build and start services

```bash
docker compose up --build
```

### 3. Reload the price model

Once the container is running:
```bash
curl -X POST http://localhost:8000/admin/reload_price_model

```

### 4. Access the API
```bash
http://localhost:8000
```

---

## API Endpoints

### 1. Health Check
```http
POST /health
```
Returns server health status.

**Example**
```bash
POST http://localhost:8000/health

```

**Response:**
```json
{
  "status": "healthy"
}
```
---

### 2. Price Model Health
```http
GET /health/price_model
```
Checks if the price prediction model is loaded and ready.

**Example**
```bash
GET http://localhost:8000/health/price_model

```

**Response:**
```json
{
  "ready": true,
  "path": "models/price_quantiles_v3"
}
```
---

### 3. Price Prediction (POST – JSON Body)
```http
POST /price
Content-Type: application/json
```

**Example**
```bash
POST http://localhost:8000/price

```

**Request Body:**
```json
{
  "make": "Toyota",
  "model": "Yaris",
  "year": 2012,
  "mileage_km_num": 30000

}

```

**Response:**
```json
{
    "green_low": 197000,
    "green_median": 202000,
    "green_high": 206000,
    "yellow": 224000,
    "red_low": 247000,
    "red_median": 256000,
    "red_high": 265000,
    "estimate_basis": "based_on_comparable_listings",
    "confidence": 0.6,
    "sample_size": 8,
    "bandwidth_clamped": false
}
```
---

### 4. Price Graph
```http
POST /price_graph
Content-Type: application/json
```

**Request Body:** Same as `/price` endpoint

**Response:** PNG image (image/png) showing a bar chart with green, yellow, and red price bands

---

### 5. Depreciation Estimation
```http
POST /depreciation
Content-Type: application/json
```

**Request Body:**
```json
{"make": "Toyota","model": "Yaris","year": 2019,"horizon_years": 6,"mileage_km_num": 30000.0}
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
---

### 6. Test Depreciation
```http
GET /test_dep
```
Tests if the depreciation endpoint is ready.


---

## Manual Setup (Local Environment – Optional)
  ⚠️ This method is intended for development, debugging, or environments where Docker is unavailable.

### 1. Navigate to the project directory
After cloning the repository:
```bash
cd CarAI
```

### 2. Create and activate a virtual environment
After cloning the repository:
```bash
python -m venv venv
```
**Windows**
```bash
venv\Scripts\activate
```
**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_api.txt

```

### 4. Configure environment variables
Create a .env file in the project root (required for depreciation features):

```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=carpricing
MYSQL_USER=car_user
MYSQL_PASSWORD=StrongPassword123
```

### 5. Ensure model files are present in `models/price_quantiles_v3/`:
   - `q20_lgbm.pkl`
   - `q50_lgbm.pkl`
   - `q80_lgbm.pkl`
   - `feature_config.json`
   - `group_medians.csv` (optional)

### 6. Start the server
From the project root:
```bash
python entry.py
```
The API will be available at:
```bash
http://localhost:8000
```

### 7. Reload the price model (required)
In a new terminal:
```bash
curl -X POST http://localhost:8000/admin/reload_price_model
```

### 8. Test the API
You can now test endpoints such as:
   - `POST /price`
   - `POST /price_graph`
   - `POST /depreciation`
   - `GET /health`
**Example:**
```bash
http://localhost:8000/price
```

---


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


---

## Dependencies

### Core Dependencies
- `pandas==2.2.2`
- `numpy==1.26.4`
- `lightgbm==4.5.0`
- `scikit-learn==1.5.2`
- `joblib==1.4.2`
- `python-dotenv==1.0.1`
- `pyarrow==17.0.0`
- `pydantic==2.9.0`
- `mysql-connector-python==9.0.0`
- `cryptography`
- `tqdm==4.66.5`

### API Dependencies
- `fastapi[standard]`
- `uvicorn[standard]`
- `SQLAlchemy`
- `PyMySQL`
- `matplotlib` (for chart generation)

---

## Development

### Model Training

The price models are trained using scripts in the repository:
`train_model_v3.py`

### Data Processing

Various data cleaning and processing scripts:
`clean_data_v3.py`


## Error Handling

- **503 Service Unavailable**: Price model not loaded
- **400 Bad Request**: Invalid input parameters
- **500 Internal Server Error**: Model prediction or database errors, try reload model.
---

## Notes

- The price model requires all three quantile models (q20, q50, q80) to be loaded
- Depreciation estimation requires a MySQL database connection
- All prices are returned in Thai Baht (THB)
- The system normalizes categorical inputs (brand, model) to uppercase
- Missing optional fields are handled gracefully with defaults
---

## License

-

## Author

-
---
