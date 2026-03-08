import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


class FDDMLPModel(models.Model):
    """Simple MLP Model for FDD Channel Estimation."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = models.Sequential(
            [
                layers.Dense(256, activation="relu", input_shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(output_dim),  # Linear activation for regression
            ]
        )

    def call(self, x):
        return self.net(x)


class FDDCNNModel(models.Model):
    """1D-CNN Model for FDD Channel Estimation (to capture frequency correlation)."""

    def __init__(self, input_shape, output_dim):
        super().__init__()
        # input_shape: [num_rb, num_features]
        self.net = models.Sequential(
            [
                layers.Conv1D(
                    64, 3, padding="same", activation="relu", input_shape=input_shape
                ),
                layers.Conv1D(64, 3, padding="same", activation="relu"),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(output_dim),
            ]
        )

    def call(self, x):
        return self.net(x)


class FDDTransformerModel(models.Model):
    """Transformer Encoder Model for FDD Channel Estimation."""

    def __init__(self, num_rb, num_features, output_dim):
        super().__init__()
        self.num_rb = num_rb
        self.embedding = layers.Dense(64)
        self.pos_encoding = self.add_weight("pos_encoding", shape=(num_rb, 64))

        encoder_layer = layers.MultiHeadAttention(num_heads=4, key_dim=16)
        self.attention = encoder_layer
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = models.Sequential(
            [layers.Dense(128, activation="relu"), layers.Dense(64)]
        )

        self.flatten = layers.Flatten()
        self.out = layers.Dense(output_dim)

    def call(self, x):
        # x: [batch, num_rb, num_features]
        x = self.embedding(x) + self.pos_encoding
        attn_out = self.attention(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.flatten(x)
        return self.out(x)


class LightGBMWrapper:
    """Wrapper for LightGBM model."""

    def __init__(self, params=None):
        import lightgbm as lgb

        self.params = params or {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        self.model = None

    def train(self, x_train, y_train):
        import lightgbm as lgb

        # Flatten target if needed (LightGBM handles single target natively)
        # For multi-output, we can use MultiOutputRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from lightgbm import LGBMRegressor

        self.model = MultiOutputRegressor(LGBMRegressor(**self.params))
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)


def build_model(model_type, input_shape, output_dim):
    if model_type == "mlp":
        # Flatten input shape for MLP
        input_dim = np.prod(input_shape)
        return FDDMLPModel(input_dim, output_dim)
    elif model_type == "cnn":
        return FDDCNNModel(input_shape, output_dim)
    elif model_type == "transformer":
        return FDDTransformerModel(input_shape[0], input_shape[1], output_dim)
    elif model_type == "lightgbm":
        return LightGBMWrapper()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
