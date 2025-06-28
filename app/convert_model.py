import tensorflow as tf

model = tf.keras.models.load_model("trained_model/plant_disease_prediction_model.h5", compile=False)
model.save("trained_model/clean_model.keras", save_format="keras_v3")
