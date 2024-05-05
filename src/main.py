import tensorflow as tf


path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# The unique characters in the file
vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab),
    mask_token=None
)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(),
    invert=True,
    mask_token=None
)


def text_from_ids(ids):
    """
    Convert the ids to text.

    :param ids: The ids to convert.
    :return: The text.
    """
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# The maximum length sentence we want for a single input in characters
ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
id_dataset = tf.data.Dataset.from_tensor_slices(ids)

sequence_length = 100
sequences = id_dataset.batch(sequence_length + 1, drop_remainder=True)


def split_input_target(sequence):
    """
    Split the input and target text from the sequence.

    :param sequence: The sequence of text.
    :return: The input text and the target text.
    :since: 1.0.0
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# Map the function to the dataset
dataset = sequences.map(split_input_target)

"""
Basic preprocessing for the dataset.

:since: 1.0.0
"""
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


class Model(tf.keras.Model):
    """
    Model class for the RNN.

    :param vocab_size: The length of the vocabulary in the StringLookup Layer.
    :param embedding_dim: The embedding dimension.
    :param rnn_units: The number of RNN units.
    :since: 1.0.0
    """

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

        def call(self, inputs, states=None, return_state=False, training=False):
            """
            Call method for the model.
            This method is used to call the model with the inputs.

            :param self: The model object.
            :param inputs: The input text.
            :param states: The initial states.
            :param return_state: The flag to return the state.
            :param training: The flag to indicate the training mode.
            :return: The output text.
            """
            x = inputs
            x = self.embedding(x, training=training)

            if states is None:
                states = self.gru.get_initial_state(x)

            x, states = self.gru(x, initial_state=states, training=training)
            x = self.dense(x, training=training)

            if return_state:
                return x, states
            else:
                return x


"""
Create the model.

:param vocab_size: The length of the vocabulary in the StringLookup Layer.
:param embedding_dim: The embedding dimension.
:param rnn_units: The number of RNN units.
:since: 1.0.0
"""
model = Model(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)
