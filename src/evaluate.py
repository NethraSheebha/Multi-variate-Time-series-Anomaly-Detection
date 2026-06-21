import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

@tf.keras.utils.register_keras_serializable(package="Custom", name="Sampling")
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def apply_dynamic_thresholding(test_errors, lookback_period=60, sigma_multiplier=4.0): # Tightened for higher precision
    y_pred_dynamic = np.zeros_like(test_errors)
    for i in range(len(test_errors)):
        historical_slice = test_errors[:i+1] if i < lookback_period else test_errors[i - lookback_period : i]
        rolling_mean = np.mean(historical_slice)
        rolling_std = np.std(historical_slice)
        dynamic_limit = rolling_mean + (sigma_multiplier * rolling_std)
        if test_errors[i] > dynamic_limit:
            y_pred_dynamic[i] = 1
    return y_pred_dynamic

def apply_persistence_filter(y_pred_raw, consecutive_minutes=4): # Upgraded time persistence window
    filtered_pred = np.zeros_like(y_pred_raw)
    counter = 0
    for i in range(len(y_pred_raw)):
        if y_pred_raw[i] == 1:
            counter += 1
            if counter >= consecutive_minutes:
                filtered_pred[i] = 1
        else:
            counter = 0
    return filtered_pred

def adjust_predictions(y_true, y_pred):
    adjusted_pred = np.copy(y_pred)
    in_anomaly = False
    start_idx = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if not in_anomaly:
                in_anomaly = True
                start_idx = i
        else:
            if in_anomaly:
                in_anomaly = False
                if np.sum(y_pred[start_idx:i]) > 0:
                    adjusted_pred[start_idx:i] = 1
    if in_anomaly and np.sum(y_pred[start_idx:]) > 0:
        adjusted_pred[start_idx:] = 1
    return adjusted_pred

def run_offline_test():
    print("⏳ [Testing Phase] Launching Recurrent OmniAnomaly Testing Loop...")
    scaler = joblib.load("models/scaler.pkl")
    
    custom_map = {"Sampling": Sampling, "Custom>Sampling": Sampling}
    with tf.keras.utils.custom_object_scope(custom_map):
        encoder = tf.keras.models.load_model("models/vae_encoder.keras", compile=False)
        decoder = tf.keras.models.load_model("models/vae_decoder.keras", compile=False)

    test_raw = np.loadtxt("data/processed/demo_stream.csv", delimiter=",")
    test_labels = np.loadtxt("data/processed/demo_labels.csv", delimiter=",")

    window_size = 30 
    test_scaled = scaler.transform(test_raw)
    
    # 🎯 FIX: Generate 3D sequence arrays matching OmniAnomaly parameters
    X_test = []
    for i in range(len(test_scaled) - window_size + 1):
        X_test.append(test_scaled[i : i + window_size])
    X_test = np.array(X_test)
    y_true = test_labels[window_size - 1 :]

    print("🚀 Running sequential inference across recurrent states...")
    _, _, z = encoder.predict(X_test, verbose=0)
    reconstructions = decoder.predict(z, verbose=0)
    
    # 🎯 FIX: Calculate MSE error summing across both spatial features and time dimensions
    test_errors = np.mean(np.square(X_test - reconstructions), axis=(1, 2))

    y_pred_raw = apply_dynamic_thresholding(test_errors, lookback_period=60, sigma_multiplier=4.0)
    y_pred_filtered = apply_persistence_filter(y_pred_raw, consecutive_minutes=4)
    y_pred_adjusted = adjust_predictions(y_true, y_pred_filtered)

    print("\n📊 --- OMNIANOMALY HIGH-PRECISION METRICS REPORT ---")
    print(classification_report(y_true, y_pred_adjusted, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_true, y_pred_adjusted)
    print("[Confusion Matrix Output]")
    print(f" └── True Negatives  : {cm[0][0]}")
    print(f" └── False Positives : {cm[0][1]}")
    print(f" └── False Negatives : {cm[1][0]}")
    print(f" └── True Positives  : {cm[1][1]}")

if __name__ == "__main__":
    run_offline_test()
