import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

conv_configs = [(1, 16, 3, 1, 1)] 
fc_configs = [(16 * 28 * 28, 10)]  
model = nn.models.Model_CNN(conv_configs=conv_configs, fc_configs=fc_configs)
model.load_model(r'best_models/model_cnn/best_model.pickle')

test_images_path = r'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))