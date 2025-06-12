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
    train_images = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_mnist()
    print('Train images:', tr_imgs.shape)
    print('Train labels:', tr_lbls.shape)
    print('Test images:', te_imgs.shape)
    print('Test labels:', te_lbls.shape)
