import tensorflow as tf
from tensorflow import keras
layers = keras.layers
# from tensorflow.keras import layers
import keras_nlp
import numpy as np


class RNN_TRANSFORMER(keras.Model):
    def __init__(self):
        super().__init__()
        self.dropout = layers.Dropout(0.2)
        self.init = layers.Dense(256, activation='relu')
        self.position_emb = keras_nlp.layers.PositionEmbedding(sequence_length=30)

        self.rnns = [layers.Bidirectional(layers.LSTM(
            units=256,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True,
        )) for _ in range(2)]

        self.enc = [keras_nlp.layers.TransformerEncoder(1024, 8) for _ in range(2)]
        self.linear_proj = [layers.Dense(26, activation='relu') for _ in range(3)]

    def call(self, x):
        # Initial layers
        x = self.dropout(self.init(x))
        x = self.position_emb(x)

        # LSTM layers forward
        for l in self.rnns:
            x = l(x)

        # Transformer Encoder forward
        for l in self.enc:
            x = l(x)

        # Prediction
        outputs = []
        for proj in self.linear_proj:
            outputs.append(proj(x))
        return tf.stack(outputs)


if __name__ == "__main__":
    model = RNN_TRANSFORMER()
    model.compile()

    output = model(np.random.randn(32, 30, 52))

    print(output.shape, '[rotor, batch, seq, features]')
    model.summary()

