import pytorch_lightning as pl
import h5py
import os
import tempfile
from torch.utils.data import DataLoader
from PySmartDL import SmartDL

class HDF5DataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.temp_dir = tempfile.mkdtemp(dir=os.getenv('global_scratch', "."))
        self.hdf5_path = os.path.join(self.temp_dir, 'data.h5')

    def prepare_data(self):
        # This method is used to perform operations that might write to the file system, such as downloading a dataset

        URLs= ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
        # Download the dataset straight to h5 file
        # Download the dataset
        for url in URLs:
            with SmartDL(url, dest=self.temp_dir,progress_bar=False) as dl:
                dl.download()
                dl.wait()
                # Unpack the dataset
                self.data = dl.get_dest()
                print(f'Dataset downloaded to {self.data}')
        # Save data to HDF5


        with h5py.File(self.hdf5_path, 'w') as f:
            f.create_dataset('data', data=self.data)

    def setup(self, stage=None):
        #This is used to change the store of data into a meaningful Dataset class that usually implements the __getitem__ and __len__ methods,
        #or an IterableDataset class that implements the __iter__ method.
        # This method is called after prepare_data and is used to perform operations that might require data to be loaded in memory
        with h5py.File(self.hdf5_path, 'r') as f:
            self.data = f['data'][:]
        print(f'Dataset loaded from {self.hdf5_path}')
    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def teardown(self, stage=None):
        os.remove(self.hdf5_path)
        os.rmdir(self.temp_dir)