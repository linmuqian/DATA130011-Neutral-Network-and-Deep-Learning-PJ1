# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

def visualize_mlp_weights(model, save_dir):
    layer_index = 0
    for layer in model.layers:
        if hasattr(layer, 'params') and 'W' in layer.params:
            weights = layer.params['W']
            num_neurons = weights.shape[1]
            rows = int(np.ceil(np.sqrt(num_neurons)))
            cols = int(np.ceil(num_neurons / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten()
            for i in range(num_neurons):
                weight_vector = weights[:, i]
                vector_length = len(weight_vector)
                side_length = int(np.sqrt(vector_length))
                if side_length * side_length == vector_length:
                    weight_img = weight_vector.reshape(side_length, side_length)
                else:
                    new_size = int(np.ceil(np.sqrt(vector_length)))
                    new_vector = np.pad(weight_vector, (0, new_size * new_size - vector_length), mode='constant')
                    weight_img = new_vector.reshape(new_size, new_size)
                axes[i].matshow(weight_img, cmap='bwr')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            
            # Hide any unused subplots
            for j in range(num_neurons, rows * cols):
                axes[j].axis('off')
            fig.suptitle(f"MLP Layer {layer_index+1} Weights")
            save_path = f"{save_dir}/mlp_layer_{layer_index+1}_weights.png"
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Successfully saved {save_path}")
            layer_index += 1
        else:
            print(f"Layer {layer_index+1} does not have weights to visualize.")


def visualize_model_weights(model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(model, nn.models.Model_MLP):
        visualize_mlp_weights(model, save_dir)

    else:
        print("Unsupported model type.")


model = nn.models.Model_MLP()
model.load_model(r'best_models/model_mlp_5/best_model.pickle')

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

visualize_model_weights(model, r"best_models/model_mlp_5")
    