import os
import joblib
import numpy as np
import tensorflow as tf
from keras import layers, models, losses
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("OmniAnomaly_3D_Sequence_Anomaly_Detection")

tf.random.set_seed(42)
np.random.seed(42)

# --- 1. OMNIANOMALY 3D SEQUENCE PREPROCESSING ---
def preprocess_and_sequence_data(train_path, val_path, window_size=30):
    print("⏳ Loading and formatting temporal sequences for OmniAnomaly...")
    train_raw = np.loadtxt(train_path, delimiter=',')
    val_raw = np.loadtxt(val_path, delimiter=',')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # 🎯 FIX: Leave the windows as 3D structures (Total_Windows, 30_Timesteps, 38_Features)
    # instead of flattening them. This is what the recurrent GRU layer requires.
    def create_recurrent_windows(data, size):
        windows = []
        for i in range(len(data) - size + 1):
            windows.append(data[i : i + size])
        return np.array(windows)
        
    X_train = create_recurrent_windows(train_scaled, window_size)
    X_val = create_recurrent_windows(val_scaled, window_size)
    return X_train, X_val

# --- 2. THE STOCHASTIC RECURRENT SAMPLING LAYER ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="Sampling")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- 3. OMNIANOMALY RECURRENT NETWORK ---
class OmniAnomalyVAE(models.Model):
    def __init__(self, timesteps, features, latent_dim=16):
        super(OmniAnomalyVAE, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        
        # 🎯 RECURRENT ENCODER (OmniAnomaly Spec)
        encoder_inputs = layers.Input(shape=(timesteps, features))
        # GRU captures step-by-step sequential dependencies over time
        x = layers.GRU(128, return_sequences=False, activation="tanh")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
        # 🎯 RECURRENT DECODER (OmniAnomaly Spec)
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation="relu")(decoder_inputs)
        # Repeat the latent code vector across all lookback timesteps
        x = layers.RepeatVector(timesteps)(x)
        # Recurrent GRU tracks generation trajectories backward
        x = layers.GRU(128, return_sequences=True, activation="tanh")(x)
        decoder_outputs = layers.TimeDistributed(layers.Dense(features, activation="sigmoid"))(x)
        self.decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            recon_loss = tf.reduce_mean(losses.mean_squared_error(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + (0.02 * kl_loss) 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "recon_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

if __name__ == "__main__":
    # Ingest data as 3D array spaces
    X_train, X_val = preprocess_and_sequence_data("data/processed/train.csv", "data/processed/validation.csv", window_size=30)
    
    _, timesteps, features = X_train.shape # Extract dimensions (30, 38)
    print(f"📊 OmniAnomaly Data Input Matrix Configuration: {X_train.shape}")
    
    # 🎯 START MLFLOW AUTO-LOGGING RUN
    with mlflow.start_run(run_name="OmniAnomaly_GRU_Training") as run:
        
        # Log Hyperparameters to MLflow tracking panel
        mlflow.log_param("window_size", 30)
        mlflow.log_param("latent_dim", 16)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("epochs", 15)

        vae = OmniAnomalyVAE(timesteps=timesteps, features=features, latent_dim=16)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        # Train the model while MLflow tracks internal Keras history loops
        print("🚀 Training OmniAnomaly Model with MLflow Tracking...")
        history = vae.fit(X_train, epochs=15, batch_size=128, verbose=1)
        
        # Log final metric checkpoints
        mlflow.log_metric("final_total_loss", float(history.history['loss'][-1]))
        mlflow.log_metric("final_recon_loss", float(history.history['recon_loss'][-1]))
        mlflow.log_metric("final_kl_loss", float(history.history['kl_loss'][-1]))
        
        # Calculate validation threshold settings
        reconstructions = vae.predict(X_val)
        val_errors = np.mean(np.square(X_val - reconstructions), axis=(1, 2))
        threshold = np.percentile(val_errors, 99.5)
        print(f"\n🎯 Engineered Threshold: {threshold:.6f}")
        
        mlflow.log_metric("calibrated_threshold", float(threshold))
        np.save("models/threshold.npy", threshold)

        sample_in = X_train[:5]
        
        # Encoder Signature (Input: 3D Sequence -> Output: [z_mean, z_log_var, z])
        encoder_out = vae.encoder.predict(sample_in)
        encoder_signature = infer_signature(sample_in, encoder_out)
        
        # Decoder Signature (Input: 2D Latent Vector -> Output: 3D Reconstruction)
        decoder_in = encoder_out[2]  # Takes the sampled z vector
        decoder_out = vae.decoder.predict(decoder_in)
        decoder_signature = infer_signature(decoder_in, decoder_out)

        # -------------------------------------------------------------
        # 🎯 REGISTER MODELS WITH SIGNATURES
        # -------------------------------------------------------------
        print("💾 Registering model components into MLflow Artifact Server...")
        
        mlflow.keras.log_model(
            vae.encoder, 
            name="omnianomaly_encoder",
            signature=encoder_signature,
            registered_model_name="OmniAnomaly_Encoder",
        )
        mlflow.keras.log_model(
            vae.decoder, 
            name="omnianomaly_decoder",
            signature=decoder_signature,
            registered_model_name="OmniAnomaly_Decoder",
        )
        
        vae.encoder.save("models/vae_encoder.keras")
        vae.decoder.save("models/vae_decoder.keras")
        # Log static scaler parameter configuration assets
        mlflow.log_artifact("models/scaler.pkl", artifact_path="preprocessing")
        mlflow.log_artifact("models/threshold.npy", artifact_path="alerting_parameters")
        
        print("🎉 Run logged completely. Weight binaries version-locked in MLflow UI.")