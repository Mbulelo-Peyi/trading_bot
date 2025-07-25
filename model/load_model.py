from tensorflow import keras
from model.transformer_model import PositionalEncoding, TransformerBlock

def load_model():
    return keras.models.load_model(
        "hybrid_trading_model.keras",
        custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "TransformerBlock": TransformerBlock
        }
    )
