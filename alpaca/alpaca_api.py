import os
import alpaca_trade_api as tradeapi
import dotenv

dotenv.load_dotenv()

# Load credentials
API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = os.getenv("ALPACA_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_KEY or ALPACA_SECRET in environment.")

# Alpaca REST API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


def get_account_info():
    """Return account cash, equity, and buying power."""
    try:
        account = api.get_account()
        return {
            "cash": account.cash,
            "buying_power": account.buying_power,
            "equity": account.equity,
            "status": account.status
        }
    except Exception as e:
        print(f"Failed to fetch account info: {e}")
        return {"error": str(e)}


def get_positions():
    """Return all open positions."""
    try:
        positions = api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "unrealized_pl": p.unrealized_pl,
            } for p in positions
        ]
    except Exception as e:
        print(f"Failed to fetch positions: {e}")
        return {"error": str(e)}


def get_blotter():
    """Get last 10 Alpaca orders."""
    try:
        orders = api.list_orders(status='all', limit=10)
        blotter_lines = []

        for order in orders:
            line = f"{order.side.capitalize()} {order.qty} {order.symbol} @ {order.limit_price or 'market'} ({order.status})"
            if order.filled_at:
                filled_price = order.filled_avg_price or "N/A"
                line += f", filled @ {filled_price}"
            blotter_lines.append(line)

        return blotter_lines
    except Exception as e:
        print(f"Error fetching blotter: {e}")
        return []


def execute_trade(symbol, qty, side, type='market', limit_price=None):
    """
    Place an order with Alpaca.
    type: 'market' or 'limit'
    side: 'buy' or 'sell'
    """
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force='gtc',
            limit_price=limit_price if type == 'limit' else None
        )
        print(f"Trade executed: {order.id}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        raise


def cancel_all_orders():
    """Cancel all open orders."""
    try:
        api.cancel_all_orders()
        print("All orders canceled.")
        return {"message": "All open orders have been canceled."}
    except Exception as e:
        print(f"Failed to cancel orders: {e}")
        return {"error": str(e)}

    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=type,
        time_in_force='gtc'
    )
    return order