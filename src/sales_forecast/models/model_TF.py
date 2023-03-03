import logging
import tensorflow as tf
from typing import Optional


__all__ = ["transformer_model_architecture"]

logger = logging.getLogger()


N_BLOCKS = 6
SKIP_CONNECTION_STRENGTH = 0.9
OUTPUT_SEQUENCE_LENGTH = 27

N_HEADS = 3
EMBED_DIM = 64
FF_DIM = 256
DROPOUT_RATE = 0.1


def get_model_TF(
    input_shape,
    n_blocks: Optional[int] = None,
    skip_connection_strength: Optional[float] = None,
    output_sequence_length: Optional[int] = None,
):
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if n_blocks is None:
        n_blocks = N_BLOCKS
    if skip_connection_strength is None:
        skip_connection_strength = SKIP_CONNECTION_STRENGTH

    encoder_inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoder_inputs)
    for k in range(n_blocks):
        x_old = x
        transformer_block = TransformerBlock(input_shape[-1])
        x = transformer_block(x)
        x = (1.0 - skip_connection_strength) * x + skip_connection_strength * x_old

    decoder_inputs = tf.keras.layers.Reshape((output_sequence_length, -1))(x)
    decoder_outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(input_shape[-1])
    )(decoder_inputs)

    model = tf.keras.models.Model(encoder_inputs, decoder_outputs)

    return model


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_features: int,
        n_heads: Optional[int] = None,
        embed_dim: Optional[int] = None,
        ff_dim: Optional[int] = None,
        dropout_rate: Optional[int] = None,
    ):
        if n_heads is None:
            n_heads = N_HEADS
        if embed_dim is None:
            embed_dim = EMBED_DIM
        if ff_dim is None:
            ff_dim = FF_DIM
        if dropout_rate is None:
            dropout_rate = DROPOUT_RATE

        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="gelu"),
                tf.keras.layers.Dense(n_features),
            ]
        )
        self.layernorm_1 = tf.keras.layers.BatchNormalization()
        self.layernorm_2 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_1(attn_output, training=training)
        out_1 = self.layernorm_1(inputs + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output, training=training)

        return self.layernorm_2(out_1 + ffn_output)
