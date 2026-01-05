# How the Car Pricing System Works: Logic and Math Explained

This document explains the logic and mathematics behind the car pricing prediction and depreciation models in simple, easy-to-understand terms.

---

## Table of Contents
1. [Price Prediction Model](#price-prediction-model)
2. [Depreciation Model](#depreciation-model)
3. [How Data is Stored](#how-data-is-stored)
4. [The Complete Flow](#the-complete-flow)

---

## Price Prediction Model

### What Does It Do?
The price prediction model takes information about a car (brand, model, year, mileage, etc.) and predicts what price range it should sell for in the Thai market. Instead of giving one exact price, it gives you three price bands to help with selling strategy.

### The Machine Learning Approach: Quantile Regression

**What is Quantile Regression?**
Think of it like this: instead of predicting the "average" price, we predict three different price points:
- **q20**: The price where 20% of similar cars sell for less (a low price)
- **q50**: The median price where 50% of similar cars sell for less (the middle price)
- **q80**: The price where 80% of similar cars sell for less (a high price)

It's like asking: "What's the lowest reasonable price? What's the middle price? What's the highest reasonable price?"

**Why Use Three Models?**
We actually train **three separate LightGBM models**:
- One model learns to predict q20 (low prices)
- One model learns to predict q50 (median prices)
- One model learns to predict q80 (high prices)

Each model is trained to be good at predicting its specific price point. This gives us a price range, not just one number.

### The Math Behind It

#### Step 1: Log Transformation
**Why?** Car prices can vary wildly (100,000 THB to 5,000,000 THB). Using log(price) makes the numbers more manageable and helps the model learn better patterns.

**The Formula:**
```
log_price = log(actual_price)
```

**Example:**
- A 500,000 THB car: log(500,000) ≈ 13.12
- A 1,000,000 THB car: log(1,000,000) ≈ 13.82

The difference between 13.12 and 13.82 is much smaller than 500,000 and 1,000,000, making it easier for the model to learn.

#### Step 2: Model Prediction
Each of the three models (q20, q50, q80) predicts a log(price) value:
```
log_q20 = q20_model.predict(car_features)
log_q50 = q50_model.predict(car_features)
log_q80 = q80_model.predict(car_features)
```

#### Step 3: Convert Back to Real Prices
We use the exponential function (reverse of log) to get actual prices:
```
q20 = exp(log_q20)
q50 = exp(log_q50)
q80 = exp(log_q80)
```

**Example:**
- If log_q50 = 13.12, then q50 = exp(13.12) ≈ 500,000 THB

### Feature Engineering (Making the Data Useful)

The model doesn't just use raw data. It creates "engineered features" that help it understand patterns:

**Basic Features:**
- `brand`: Car manufacturer (TOYOTA, HONDA, etc.)
- `model`: Car model name (Yaris, Civic, etc.)
- `year`: Manufacturing year
- `mileage_km_num`: How many kilometers the car has driven

**Engineered Features (Created from Basic Data):**
- `age`: Current year - car year (how old the car is)
  - Example: 2024 - 2019 = 5 years old

- `log_mileage`: log(mileage + 1)
  - Why? Mileage can be huge (100,000 km), so we compress it
  - The +1 prevents errors when mileage is 0

- `sqrt_mileage`: √mileage
  - Another way to compress large mileage numbers

- `mileage_per_year`: mileage ÷ age
  - Tells us: does this car drive a lot or a little per year?
  - Example: 50,000 km ÷ 5 years = 10,000 km/year

- `age_x_mileage`: age × mileage
  - Captures the interaction: older cars with high mileage are worth less

- `mileage_per_age`: Same as mileage_per_year (alternative calculation)

### Post-Processing: Making Predictions Realistic

After the models predict prices, we apply "sanity checks" to make sure the predictions make sense:

#### 1. Blending with Group Medians
**What?** We look up the historical median price for this brand/model/year combination from our database.

**Why?** Sometimes the model might predict something weird. Blending helps anchor it to reality.

**The Math:**
```
final_q50 = (1 - blend_weight) × model_q50 + blend_weight × group_median
```

**Example:**
- Model predicts: 520,000 THB
- Historical median: 500,000 THB
- Blend weight: 30% (0.30)
- Final: 0.70 × 520,000 + 0.30 × 500,000 = 514,000 THB

#### 2. Sanity Caps
We make sure the price bands don't get too extreme:

**Rules:**
- q20 cannot be less than 65% of q50 (low price can't be too low)
- q80 cannot be more than 175% of q50 (high price can't be too high)

**The Math:**
```
q20 = max(q20, q50 × 0.65)
q80 = min(q80, q50 × 1.75)
```

**Example:**
- If q50 = 500,000 THB
- q20 must be at least: 500,000 × 0.65 = 325,000 THB
- q80 must be at most: 500,000 × 1.75 = 875,000 THB

#### 3. Re-anchoring to Group Median
If we have a group median, we also make sure q50 stays within ±35% of it:

```
q50 = clip(q50, 0.65 × group_median, 1.35 × group_median)
```

This prevents the model from predicting prices that are way off from historical data.

### Creating Price Bands (Green/Yellow/Red)

Once we have q50 (the median price), we create three selling strategy bands:

**Green Band (Sell Fast):** 8-12% below median
- `green_low` = q50 × 0.88 (12% below)
- `green_median` = q50 × 0.90 (10% below)
- `green_high` = q50 × 0.92 (8% below)

**Yellow Band (Market Median):** Exactly at q50
- `yellow` = q50

**Red Band (Hold Out):** 10-18% above median
- `red_low` = q50 × 1.10 (10% above)
- `red_median` = q50 × 1.14 (14% above)
- `red_high` = q50 × 1.18 (18% above)

**Example:**
If q50 = 500,000 THB:
- Green: 440,000 - 460,000 THB (sell quickly)
- Yellow: 500,000 THB (fair market price)
- Red: 550,000 - 590,000 THB (wait for the right buyer)

### Price Rounding

Finally, we round prices to make them realistic:
- If price < 1,000,000 THB: round to nearest 1,000 THB
- If price ≥ 1,000,000 THB: round to nearest 5,000 THB

**Example:**
- 523,456 THB → 523,000 THB
- 1,234,567 THB → 1,235,000 THB

---

## Depreciation Model

### What Does It Do?
The depreciation model predicts how much a car will be worth in the future. It answers: "If I buy this car today, what will it be worth in 1 year? 2 years? 5 years?"

### The Mathematical Model: Log-Linear Regression

**The Core Formula:**
```
log(price) = a × age + b × log(mileage + 1) + c
```

Where:
- `age` = current_year - car_year (how old the car is)
- `mileage` = kilometers driven
- `a`, `b`, `c` = numbers the model learns from data

**Why This Formula?**
1. **Age makes cars cheaper:** As a car gets older, its value goes down
2. **Mileage makes cars cheaper:** More kilometers = lower value
3. **Log transformation:** Just like in price prediction, we use logs to handle the wide range of prices

### How It Works

#### Step 1: Find Similar Cars (Cohort)
The model looks in the database for all cars with the same:
- Brand (e.g., TOYOTA)
- Model (e.g., Yaris)
- Optionally: Submodel (e.g., 1.5 E)

This group of similar cars is called a "cohort."

**Why?** We need at least 30 similar cars to build a reliable model. If we don't have enough, we relax the requirements (maybe ignore submodel).

#### Step 2: Clean the Data
We remove outliers (extreme prices) that might confuse the model:
- Remove the top 5% and bottom 5% of prices
- This is called "winsorization"

**Example:**
If we have prices: [200k, 300k, 400k, 500k, 600k, 700k, 800k, 900k, 1M, 2M]
- Remove 2M (top 5%)
- Remove 200k (bottom 5%)
- Use the rest for training

#### Step 3: Train the Model
We use **HuberRegressor**, which is a "robust" regression method. "Robust" means it doesn't get confused by a few weird data points.

**What the Model Learns:**
- How much does price drop per year of age? (coefficient `a`)
- How much does price drop per unit of log(mileage)? (coefficient `b`)
- What's the base price? (intercept `c`)

**Example Output:**
- `a = -0.08` means: each year of age reduces log(price) by 0.08
- `b = -0.15` means: each unit of log(mileage) reduces log(price) by 0.15

#### Step 4: Calculate Uncertainty
The model also calculates how "uncertain" its predictions are:

```
sigma_log = standard_deviation of prediction errors
```

**What This Means:**
- If sigma_log = 0.25, then predictions are typically ±25% accurate
- We use this to create "confidence bands" (lower and upper bounds)

**The Math:**
```
predicted_price = exp(model_prediction)
lower_bound = exp(model_prediction - sigma_log)
upper_bound = exp(model_prediction + sigma_log)
```

**Example:**
- Model predicts: log(price) = 13.12 → price = 500,000 THB
- sigma_log = 0.25
- Lower: exp(13.12 - 0.25) = 400,000 THB
- Upper: exp(13.12 + 0.25) = 625,000 THB
- So we're 68% confident the price is between 400k and 625k THB

#### Step 5: Estimate Future Mileage
To predict future prices, we need to estimate how many kilometers the car will drive each year.

**How?** We calculate the median kilometers per year from the cohort:
```
km_per_year = median(mileage ÷ age) for all cars in cohort
```

**Example:**
- Car 1: 50,000 km ÷ 5 years = 10,000 km/year
- Car 2: 60,000 km ÷ 4 years = 15,000 km/year
- Car 3: 48,000 km ÷ 6 years = 8,000 km/year
- Median: 10,000 km/year

If we can't calculate this, we use a default: 12,000 km/year

#### Step 6: Project Future Prices
For each future year (1, 2, 3, 4, 5 years ahead):

1. **Calculate new age:**
   ```
   future_age = current_age + years_ahead
   ```

2. **Calculate new mileage:**
   ```
   future_mileage = current_mileage + (km_per_year × years_ahead)
   ```

3. **Predict price:**
   ```
   log_price = model.predict([future_age, log(future_mileage + 1)])
   future_price = exp(log_price)
   ```

4. **Calculate confidence bands:**
   ```
   lower = exp(log_price - sigma_log)
   upper = exp(log_price + sigma_log)
   ```

5. **Calculate depreciation percentage:**
   ```
   depreciation_pct = (current_price - future_price) ÷ current_price × 100
   ```

**Example:**
- Current: 500,000 THB, age = 5, mileage = 50,000 km
- In 1 year: age = 6, mileage = 62,000 km (assuming 12k km/year)
- Predicted price: 460,000 THB
- Depreciation: (500,000 - 460,000) ÷ 500,000 × 100 = 8%

---

## How Data is Stored

### Price Prediction Models

**Location:** `models/price_quantiles_v3/`

**Files:**
1. **`q20_lgbm.pkl`** - The model that predicts low prices (20th percentile)
2. **`q50_lgbm.pkl`** - The model that predicts median prices (50th percentile)
3. **`q80_lgbm.pkl`** - The model that predicts high prices (80th percentile)
4. **`feature_config.json`** - Configuration file that tells the system:
   - Which features to use
   - Which are categorical (brand, model) vs numeric (year, mileage)
   - The blend weight (30% = 0.30)
   - Model version information
5. **`group_medians.csv`** - A lookup table with historical median prices for brand/model/year combinations
6. **`metrics.json`** - Performance metrics from when the model was trained

**Format:**
- `.pkl` files are Python "pickle" files - they store the trained LightGBM models in binary format
- `.json` files are human-readable configuration
- `.csv` files are comma-separated data tables

**How Models Are Loaded:**
When the API starts, it calls `load_artifacts()` which:
1. Loads the three `.pkl` files into memory (variables `_q20`, `_q50`, `_q80`)
2. Reads `feature_config.json` to know what features to expect
3. Loads `group_medians.csv` into a pandas DataFrame for quick lookups

**Why This Format?**
- Models are large binary files - pickle format is efficient
- Configuration is in JSON so humans can read/edit it
- Group medians in CSV are easy to update with new data

### Depreciation Data

**Location:** MySQL database table `car_listings_master`

**What's Stored:**
- `brand` - Car manufacturer
- `model` - Car model name
- `submodel` - Optional submodel/trim
- `year` - Manufacturing year
- `mileage` - Kilometers driven
- `price` - Selling price (in THB)
- `dateRecorded` - When the listing was recorded

**How It's Used:**
1. When you request depreciation for a car, the system queries the database
2. It finds all cars matching brand/model (and optionally submodel)
3. It uses this "cohort" to train a small regression model on-the-fly
4. The model is used to predict current and future prices

**Why Database Instead of Files?**
- Historical car listings are constantly growing
- Databases are better for querying and filtering
- Can handle millions of records efficiently
- Easy to add new listings without retraining models

### Feature Engineering Storage

**Not Stored, Calculated On-the-Fly:**
The engineered features (age, log_mileage, etc.) are **not** stored anywhere. They're calculated in real-time when you make a prediction request.

**Why?**
- They can be calculated from basic data (year, mileage)
- Storing them would waste space
- They might change if calculation logic changes

**Where Calculated:**
In the `_add_features()` function in `price_helper.py`:
1. Takes basic car info (brand, model, year, mileage)
2. Calculates age = current_year - year
3. Calculates log_mileage = log(mileage + 1)
4. Calculates all other engineered features
5. Returns a DataFrame ready for the model

---

## The Complete Flow

### Price Prediction Flow

1. **User sends request:**
   ```json
   {
     "make": "TOYOTA",
     "model": "Yaris",
     "year": 2019,
     "mileage_km_num": 30000
   }
   ```

2. **System builds features:**
   - Normalizes: "TOYOTA" → "TOYOTA" (uppercase)
   - Calculates: age = 2024 - 2019 = 5
   - Calculates: log_mileage = log(30000 + 1) ≈ 10.31
   - Calculates: mileage_per_year = 30000 ÷ 5 = 6000
   - And so on...

3. **Three models predict:**
   - q20_model → log_q20 = 12.85
   - q50_model → log_q50 = 13.12
   - q80_model → log_q80 = 13.45

4. **Convert to real prices:**
   - q20 = exp(12.85) = 380,000 THB
   - q50 = exp(13.12) = 500,000 THB
   - q80 = exp(13.45) = 690,000 THB

5. **Apply sanity checks:**
   - Look up group median: 510,000 THB
   - Blend: q50 = 0.7 × 500,000 + 0.3 × 510,000 = 503,000 THB
   - Apply caps: q20 ≥ 327,000, q80 ≤ 880,000

6. **Create price bands:**
   - Green: 442,640 - 462,760 THB
   - Yellow: 503,000 THB
   - Red: 553,300 - 593,540 THB

7. **Round prices:**
   - Green: 443,000 - 463,000 THB
   - Yellow: 503,000 THB
   - Red: 553,000 - 594,000 THB

8. **Return to user:**
   ```json
   {
     "green_low": 443000,
     "green_median": 453000,
     "green_high": 463000,
     "yellow": 503000,
     "red_low": 553000,
     "red_median": 573000,
     "red_high": 594000
   }
   ```

### Depreciation Flow

1. **User sends request:**
   ```json
   {
     "make": "TOYOTA",
     "model": "Yaris",
     "year": 2019,
     "mileage_km_num": 30000,
     "horizon_years": 5
   }
   ```

2. **System queries database:**
   - Finds all TOYOTA Yaris listings
   - Filters: year between 1990 and 2025
   - Gets: brand, model, year, mileage, price
   - Result: 150 similar cars

3. **Clean data:**
   - Remove prices < 0
   - Remove years < 1990
   - Winsorize: remove top/bottom 5%
   - Calculate age for each car

4. **Train model:**
   - X = [age, log(mileage+1)] for each car
   - y = log(price) for each car
   - Fit HuberRegressor
   - Calculate sigma_log = 0.28

5. **Calculate km/year:**
   - For each car: km_per_year = mileage ÷ age
   - Median: 12,000 km/year

6. **Predict current price:**
   - age = 2024 - 2019 = 5
   - mileage = 30,000
   - log_price = model.predict([5, log(30001)]) = 13.12
   - price = exp(13.12) = 500,000 THB
   - lower = exp(13.12 - 0.28) = 380,000 THB
   - upper = exp(13.12 + 0.28) = 660,000 THB

7. **Project future:**
   - Year 1: age=6, mileage=42,000 → price=460,000 THB
   - Year 2: age=7, mileage=54,000 → price=425,000 THB
   - Year 3: age=8, mileage=66,000 → price=395,000 THB
   - Year 4: age=9, mileage=78,000 → price=370,000 THB
   - Year 5: age=10, mileage=90,000 → price=350,000 THB

8. **Return to user:**
   ```json
   {
     "predicted_price_now": 500000.00,
     "lower_now": 380000.00,
     "upper_now": 660000.00,
     "annual_projection": [
       {"calendar_year": 2025, "price": 460000.00, ...},
       {"calendar_year": 2026, "price": 425000.00, ...},
       ...
     ]
   }
   ```

---

## Summary

**Price Prediction:**
- Uses 3 LightGBM models to predict low/median/high prices
- Works in "log space" to handle wide price ranges
- Applies sanity checks and blends with historical data
- Creates green/yellow/red price bands for selling strategy

**Depreciation:**
- Uses HuberRegressor to learn how age and mileage affect price
- Trains on-the-fly using similar cars from database
- Projects future prices by estimating future age and mileage
- Provides confidence bands showing uncertainty

**Storage:**
- Models stored as `.pkl` files (binary)
- Configuration as `.json` files (readable)
- Historical data in MySQL database (queryable)
- Features calculated on-the-fly (not stored)

Both systems use log transformations to handle the wide range of car prices, and both provide ranges/bands rather than single point estimates, which is more realistic and useful for decision-making.

