# HDF5Dataloader

## Overview

HDF5Dataloader is a template PyTorch dataloader designed to efficiently handle datasets by storing them in HDF5 files. This approach enhances the speed of loading data between HPC memory and disk, avoiding the creation of numerous small files.

## Features

- Converts datasets to HDF5 format
- Faster data loading for HPC environments
- Reduces the number of small files

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset and convert it to HDF5 format using the provided script.
2. Use the HDF5Dataloader in your PyTorch training loop.

Example:
```python
from hdf5dataloader import HDF5Dataloader

# Initialize the dataloader
dataloader = HDF5Dataloader(hdf5_file='path_to_hdf5_file.h5', batch_size=32)

# Iterate through the dataset
for data in dataloader:
    # Your training code here
    pass
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.