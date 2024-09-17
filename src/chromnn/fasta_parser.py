import numpy as np
import os


def one_hot_encode_sequence(sequence):
    """One-hot encode a single DNA sequence."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence):
        index = mapping.get(nucleotide.upper(), 4)
        if index < 4:
            one_hot[i, index] = 1.0
    return one_hot


def parse_fasta(file_path):
    """Parse a multi-sequence FASTA file and return one-hot encoded sequences."""
    sequences = []
    with open(file_path, "r") as f:
        sequence_id = None
        sequence = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None and sequence:
                    sequences.append(
                        (sequence_id, one_hot_encode_sequence("".join(sequence)))
                    )
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None and sequence:
            sequences.append((sequence_id, one_hot_encode_sequence("".join(sequence))))
    return sequences


def save_sequences_to_npz(sequences, output_file):
    """Save one-hot encoded sequences to a compressed .npz file."""
    data_dict = {seq_id: seq for seq_id, seq in sequences}
    np.savez_compressed(output_file, **data_dict)


def parse_fasta_and_save_npz(fasta_file, output_file):
    """Parse a FASTA file, one-hot encode the sequences, and save them as a .npz file."""
    sequences = parse_fasta(fasta_file)
    save_sequences_to_npz(sequences, output_file)
    print(f"Saved {len(sequences)} sequences to {output_file}")


def load_npz_data(npz_file):
    """Load one-hot encoded sequences from an .npz file."""
    data = np.load(npz_file)
    sequences = {}
    for key in data.files: 
        sequences[key] = data[key]
    return sequences
