import sys
from chromnn.data_prep import load_bigwig_data

if __name__ == "__main__":
    bigwig_file = sys.argv[1]
    chrom = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    data = load_bigwig_data(bigwig_file, chrom, start, end)
    print(data)