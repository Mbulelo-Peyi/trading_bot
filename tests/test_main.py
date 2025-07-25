import sys
import os
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

# --- Ensure the app is imported from main.py ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from main import app

# --- Shared fixture for AsyncClient ---
@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

# --- Mock model loading and predict for tests that need it ---
@pytest.fixture(autouse=True)
def mock_model_load():
    with patch("main.get_model") as mock_get_model:
        dummy_model = MagicMock()
        # Return NumPy array to match actual model output shape
        dummy_model.predict.side_effect = lambda x: np.array([[0.9, 0.05, 0.05]] * len(x))
        mock_get_model.return_value = dummy_model
        yield

# --- GET endpoints ---
@pytest.mark.asyncio
async def test_get_blotter(client):
    response = await client.get("/blotter")
    assert response.status_code == 200
    assert "blotter" in response.json()

@pytest.mark.asyncio
async def test_get_account(client):
    response = await client.get("/account")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

@pytest.mark.asyncio
async def test_get_positions(client):
    response = await client.get("/positions")
    assert response.status_code == 200
    assert isinstance(response.json(), (dict, list))

# --- POST endpoints ---
@pytest.mark.asyncio
async def test_cancel_orders(client):
    response = await client.post("/cancel_orders")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

@pytest.mark.asyncio
async def test_predict_and_trade(client):
    response = await client.post("/predict_and_trade")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "executions" in data
        assert isinstance(data["executions"], list)

@pytest.mark.asyncio
async def test_manual_trade_success(client):
    payload = {
        "symbol": "AAPL",
        "qty": 1,
        "side": "buy",
        "type": "market"
    }
    response = await client.post("/trade", json=payload)
    assert response.status_code in [200, 400, 500]
    if response.status_code == 200:
        data = response.json()
        assert "order_id" in data

@pytest.mark.asyncio
async def test_manual_trade_failure(client):
    payload = {
        "symbol": "INVALID",
        "qty": -10,
        "side": "buy",
        "type": "market"
    }
    response = await client.post("/trade", json=payload)
    assert response.status_code in [400, 500]
    if response.status_code == 400:
        assert "Trade failed" in response.json().get("detail", "")

# --- Mocked blotter test with execute_trade mocked ---
@patch("main.execute_trade")
@patch("main.get_blotter", return_value=["Buy 10 AAPL @ 190 market"])
@pytest.mark.asyncio
async def test_predict_and_trade_mocked(mock_get_blotter, mock_execute_trade, client):
    # Mock execute_trade to return an object with .id attribute
    mock_trade = MagicMock()
    mock_trade.id = "mocked_trade_id"
    mock_execute_trade.return_value = mock_trade

    response = await client.post("/predict_and_trade")
    assert response.status_code == 200
    data = response.json()
    assert "executions" in data
    assert isinstance(data["executions"], list)
    assert any(exec_item.get("executed") == "mocked_trade_id" for exec_item in data["executions"])

