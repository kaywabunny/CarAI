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

