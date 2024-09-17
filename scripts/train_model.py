import sys
from chromnn.train import train_model

if __name__ == "__main__":
    sequence_data = # Load sequences here (e.g., one-hot encoded sequences)
    coverage_data = # Load coverage data here (e.g., from bigWig files)

    output_model_path = sys.argv[1]
    train_model(sequence_data, coverage_data, output_model_path)