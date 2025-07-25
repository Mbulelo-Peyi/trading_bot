from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.load_model import load_model
from alpaca.alpaca_api import (
    get_blotter,
    execute_trade,
    get_account_info,
    get_positions,
    cancel_all_orders
)
import numpy as np
from tensorflow.keras.layers import TextVectorization

app = FastAPI()

# Lazy load model: only load when needed
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

# Initialize vectorizer if needed
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=40)
sample_data = ["Buy 10 AAPL @ 190 market", "Sell 5 TSLA @ 900 limit"]
vectorizer.adapt(sample_data)

@app.get("/blotter")
def read_blotter():
    return {"blotter": get_blotter()}


@app.get("/account")
def account():
    return get_account_info()


@app.get("/positions")
def positions():
    return get_positions()


@app.post("/cancel_orders")
def cancel_orders():
    return cancel_all_orders()


@app.post("/predict_and_trade")
def predict_and_trade():
    blotter = get_blotter()

    if not blotter:
        return {"executions": [], "error": "No blotter data available"}

    try:
        model = get_model()
        # If model expects raw strings directly (vectorizer inside model)
        preds = model.predict(np.array(blotter, dtype=object))
        # If model expects preprocessed inputs:
        # X = vectorizer(np.array(blotter))
        # preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    decisions = []

    # preds expected shape: (num_samples, 3)
    predicted_classes = np.argmax(preds, axis=1)

    for line, pred_probs, pred_class in zip(blotter, preds, predicted_classes):
        # Check if any prediction values are invalid
        if np.any(np.isnan(pred_probs)) or np.any(np.isinf(pred_probs)):
            decisions.append({
                "line": line,
                "prediction": pred_probs.tolist(),
                "executed": "Invalid prediction (NaN or Inf)"
            })
            continue

        if pred_class == 0:  # positive class
            try:
                parts = line.split()
                side = parts[0].lower()
                qty = int(parts[1])
                symbol = parts[2].upper()
                trade = execute_trade(symbol, qty, side)
                decisions.append({
                    "line": line,
                    "prediction": pred_probs.tolist(),
                    "executed": trade.id
                })
            except Exception as e:
                decisions.append({
                    "line": line,
                    "prediction": pred_probs.tolist(),
                    "error": str(e)
                })
        else:
            decisions.append({
                "line": line,
                "prediction": pred_probs.tolist(),
                "executed": "No action (not positive class)"
            })

    return {"executions": decisions}


# --- Manual trade endpoint ---
class TradeRequest(BaseModel):
    symbol: str
    qty: int
    side: str  # 'buy' or 'sell'
    type: str = "market"
    limit_price: float = None


@app.post("/trade")
def manual_trade(request: TradeRequest):
    try:
        trade = execute_trade(
            symbol=request.symbol.upper(),
            qty=request.qty,
            side=request.side.lower(),
            type=request.type,
            limit_price=request.limit_price
        )
        return {
            "status": "submitted",
            "symbol": request.symbol.upper(),
            "qty": request.qty,
            "side": request.side.lower(),
            "order_id": trade.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Trade failed: {e}")


# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
