import h5py
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Path to your original model
h5_path = "trained_model/plant_disease_prediction_model.h5"

# Step 1: Extract and clean the model config
with h5py.File(h5_path, "r") as f:
    config_str = f.attrs["model_config"]  # No .decode() needed
    config = json.loads(config_str)


# Step 2: Remove 'batch_shape' from InputLayer config
for layer in config["config"]["layers"]:
    if layer["class_name"] == "InputLayer":
        layer["config"].pop("batch_shape", None)

# Step 3: Rebuild and save the model
model = model_from_json(json.dumps(config))
model.load_weights(h5_path)
model.save("trained_model/clean_model.keras", save_format="keras_v3")

print("✅ Model repaired and saved as clean_model.keras")
