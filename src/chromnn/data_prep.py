import numpy as np
import pyBigWig

def one_hot_encode(sequence):
    mapping = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1], "N": [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])

def load_bigwig_data(bigwig_file, chrom, start, end):
    bw = pyBigWig.open(bigwig_file)
    data = bw.values(chrom, start, end)
    bw.close()
    return np.nan_to_num(np.array(data))

def split_data(sequences, labels, split_ratio=0.8):
    split_index = int(len(sequences) * split_ratio)
    train_sequences = sequences[:split_index]
    test_sequences = sequences[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]
    return (train_sequences, train_labels), (test_sequences, test_labels)