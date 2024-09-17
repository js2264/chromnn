import sys
from chromnn.data_prep import one_hot_encode

if __name__ == "__main__":
    sequence = sys.argv[1]
    encoded_sequence = one_hot_encode(sequence)
    print(encoded_sequence)