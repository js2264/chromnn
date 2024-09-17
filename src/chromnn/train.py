import tensorflow as tf
import numpy as np
from .models import ChromNNModel
from .fasta_parser import load_npz_data
from .data_prep import one_hot_encode, load_bigwig_data, split_data


def train_model(npz_file, coverage_data, model_output_path, batch_size=32, epochs=50):

    # Load input sequences and labels
    sequence_data = load_npz_data(npz_file)

    (train_sequences, train_labels), (test_sequences, test_labels) = split_data(
        sequence_data, coverage_data
    )

    model_builder = ChromNNModel(winsize=len(train_sequences[0]))
    model = model_builder.build_model()

    model.compile(optimizer="adam", loss="mean_squared_error")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_output_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]

    model.fit(
        train_sequences,
        train_labels,
        validation_data=(test_sequences, test_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
