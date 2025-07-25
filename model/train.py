from sklearn.model_selection import train_test_split
from model.transformer_model import build_model
from model.pipeline import text_data, structured_data, labels

# Split data
X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    text_data, structured_data, labels, test_size=0.2, random_state=42, stratify=labels.argmax(axis=1)
)

# Build model
model, vectorizer = build_model()
vectorizer.adapt(X_text_train)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    {"text_input": X_text_train, "structured_input": X_struct_train},
    y_train,
    validation_data=(
        {"text_input": X_text_val, "structured_input": X_struct_val},
        y_val
    ),
    batch_size=32,
    epochs=10
)

# Save
model.save("hybrid_trading_model.keras")