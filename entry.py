from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from depreciation import DepreciationEstimator
from price_helper import predict_price, load_artifacts, is_ready, MODELS_DIR, render_price_chart
from fastapi.responses import Response
app = FastAPI()

# --- request model for POST /price ---
class PriceRequest(BaseModel):
    make: str
    model: str
    year: int
    mileage_km_num: float
    submodel: str | None = None
    gear: str | None = None
    color: str | None = None

# --- the new endpoint: thin hub that delegates to helper ---
@app.post("/price")
def price_post(req: PriceRequest):
    if not is_ready():
        raise HTTPException(status_code=503, detail="Price model not loaded")
    return predict_price(req.dict())

@app.post("/admin/reload_price_model")
def reload_price_model():
    try:
        load_artifacts(MODELS_DIR)   # uses the path above
        return {"ok": True, "path": str(MODELS_DIR)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/price_model")
def price_model_health():
    return {"ready": is_ready(), "path": str(MODELS_DIR)}


@app.post("/price_graph")
def price_graph(req: PriceRequest):
    """
    Same body as /price:
      {
        "make": "...",
        "model": "...",
        "year": 2019,
        "mileage_km_num": 30000
      }
    Returns: PNG graph (image/png)
    """
    # Reuse the same calculation used by /price
    bands = predict_price(req.dict())

    # Nice title for the chart
    title = f"{req.year} {req.make} {req.model} • {req.mileage_km_num:,} km"

    png_bytes = render_price_chart(bands, title=title)
    return Response(content=png_bytes, media_type="image/png")
# --- request model for POST /depreciation ---

class DepreciationItem(BaseModel):
    """Request model for depreciation calculation."""
    make: str
    model: str
    year: int
    mileage_km_num: float
    submodel: Optional[str] = None
    horizon_years: int = 5
    

# Initialize the estimator once at startup (uses .env for DB config)
estimator = DepreciationEstimator()

@app.get("/test_dep")
def test_dep():
    return {"message": "Depreciation endpoint is ready"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/depreciation")
def depreciation_endpoint(item: DepreciationItem):
    """
    Calculate depreciation for a vehicle.
    
    Args:
        item: DepreciationItem containing vehicle details
    
    Returns:
        DepreciationResult as dictionary
    """
    try:
        result = api_yearly_drop(
            make=item.make, 
            model=item.model, 
            year=item.year, 
            mileage_km_num=item.mileage_km_num,
            submodel=item.submodel,
            horizon_years=item.horizon_years
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def api_yearly_drop(
    make: str, 
    model: str, 
    year: int, 
    mileage_km_num: float,
    submodel: str = None,
    horizon_years: int = 5
):
    """
    Calculate depreciation using DepreciationEstimator.
    
    Args:
        make: Car brand
        model: Car model
        year: Model year
        mileage_km_num: Current mileage in kilometers
        submodel: Optional submodel/series
        horizon_years: Number of years to project forward
    
    Returns:
        Dictionary representation of DepreciationResult
    """
    # Convert mileage to int (it's expected as int in the estimator)
    mileage = int(mileage_km_num) if mileage_km_num else None
    
    # Call the estimator
    result = estimator.estimate(
        brand=make,
        model=model,
        year=year,
        mileage=mileage,
        submodel=submodel if submodel else None,
        horizon_years=horizon_years
    )
    
    # Convert DepreciationResult dataclass to dictionary for JSON response
    return {
        "brand": result.brand,
        "model": result.model,
        "submodel": result.submodel,
        "year": result.year,
        "mileage": result.mileage,
        "sample_size": result.sample_size,
        "predicted_price_now": result.predicted_price_now,
        "lower_now": result.lower_now,
        "upper_now": result.upper_now,
        "annual_projection": result.annual_projection,
        "km_per_year_assumed": result.km_per_year_assumed,
        "notes": result.notes
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)