import os
import numpy as np
import struct

# set the path environment variable to where your kaggle.json is located
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/Desktop/projects')

from kaggle.api.kaggle_api_extended import KaggleApi

### DOWNLOAD THE DATASET USING KAGGLE API KEYS ###

# authenticate via ~/.kaggle/kaggle.json
api = KaggleApi()
api.authenticate()

# ensure data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# download only if data_dir is empty
if not os.listdir(data_dir):
    print(f"Downloading dataset into {data_dir}/")
    api.dataset_download_files(
        "hojjatk/mnist-dataset",
        path=data_dir,
        unzip=True
    )
else:
    print(f"Data already present in {data_dir}/, skipping download.")


### LOAD MNIST DATASET FROM IDX FILES ###
# The following code is from the kaagle website
# Turn the images from raw byte data to data that is actually usable for the nueral network 

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data


def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def load_mnist(data_dir='data'):
    train_images = load_images(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_labels = load_labels(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_images = load_images(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_labels = load_labels(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels


class DataLoader:
    def __init__(self):
        # Data variables
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None
    
    def load_mnist_data(self):
        # Load the data using the data_loader module
        train_images, train_labels, test_images, test_labels = load_mnist()
        
        ### DATA PREPROCESSING ###
        
        # Store the test data in class variables
        self.test_images = test_images
        self.test_labels = test_labels
        
        # Split training set into training (80%) and validation (20%) sets
        np.random.seed(42)  # For reproducibility
        n_train = len(train_images)
        indices = np.random.permutation(n_train)
        train_size = int(0.8 * n_train)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_images = train_images[train_indices]
        self.train_labels = train_labels[train_indices]
        self.val_images = train_images[val_indices]
        self.val_labels = train_labels[val_indices]
        
        # Normalize the image data (0-255 to 0-1)
        self.train_images = self.train_images.astype('float32') / 255
        self.val_images = self.val_images.astype('float32') / 255
        self.test_images = self.test_images.astype('float32') / 255
        
        # Flatten the images from 28x28 to 784-dimensional vectors
        self.train_images = self.train_images.reshape(-1, 28*28)
        self.val_images = self.val_images.reshape(-1, 28*28)
        self.test_images = self.test_images.reshape(-1, 28*28)
        
        # Convert labels to one-hot encoding
        self.train_labels = self._one_hot_encode(self.train_labels)
        self.val_labels = self._one_hot_encode(self.val_labels)
        self.test_labels = self._one_hot_encode(self.test_labels)
        
        return self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels
    
    def _one_hot_encode(self, labels):
        """Convert labels to one-hot encoding"""
        n_samples = len(labels)
        n_classes = 10  # MNIST has 10 classes (0-9)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1
        return one_hot


if __name__ == '__main__':
    data_loader = DataLoader()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data_loader.load_mnist_data()
    print('Train images:', train_images.shape)
    print('Train labels:', train_labels.shape)
    print('Val images:', val_images.shape)
    print('Val labels:', val_labels.shape)
    print('Test images:', test_images.shape)
    print('Test labels:', test_labels.shape)
