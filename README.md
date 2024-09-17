# chromnn

`chromnn` is a Python package for training and using state-of-the-art neural network algorithms, to predict multi-omics coverage track signals along chromosome-long DNA sequences. 

## Features

- Interacts with multi-omics data stored in `.momics` repositories; 
- Trains and tests an optimized neural network using experimental data; 
- Predict coverage signals for any epigenetic modality, from chromosome-long DNA sequences. 

## Installation

```bash
micromamba create -n chromnn && micromamba activate chromnn
micromamba install cudatoolkit cudnn
pip install .
```

For dev install: 

```bash
micromamba create -n chromnn && micromamba activate chromnn
micromamba install "python>=3.8" cudatoolkit cudnn
pip install -e .[dev,docs]
```