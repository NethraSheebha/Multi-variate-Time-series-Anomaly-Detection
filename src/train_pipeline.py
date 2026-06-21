import os
import joblib
import numpy as np
import tensorflow as tf
from keras import layers, models, losses
from sklearn.preprocessing import MinMaxScaler

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
    
    vae = OmniAnomalyVAE(timesteps=timesteps, features=features, latent_dim=16)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    print("🚀 Training Recurrent Stochastic OmniAnomaly Network...")
    vae.fit(X_train, epochs=15, batch_size=128, verbose=1)
    
    print("📐 Calibrating baseline validation metrics...")
    reconstructions = vae.predict(X_val)
    val_errors = np.mean(np.square(X_val - reconstructions), axis=(1, 2)) # Evaluate variance across sequences
    
    threshold = np.percentile(val_errors, 99.5)
    print(f"\n🎯 OmniAnomaly Engineered Alert Threshold: {threshold:.6f}")
    
    np.save("models/threshold.npy", threshold)
    vae.encoder.save("models/vae_encoder.keras")
    vae.decoder.save("models/vae_decoder.keras")
    print("🎉 OmniAnomaly engine artifacts written successfully to models/ folder.")
