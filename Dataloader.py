import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import gzip
from pathlib import Path



class Dataloader_from_CSV:
    def __init__(self, train_csv_path, validate_csv_path, num_classes=10):
        self.train_data = self._load_data(train_csv_path, num_classes)
        self.validate_data = self._load_data(validate_csv_path, num_classes)
    
    def _load_data(self, csv_path, num_classes):
        data_dict = {}
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                label = int(row[0])
                pixels = np.array(row[1:], dtype=np.float32) / 255.0  
                one_hot_label = self._one_hot_encode(label, num_classes)  
                data_dict[i] = (pixels, one_hot_label)  
        return data_dict
    
    def _one_hot_encode(self, label, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[label] = 1.0
        return one_hot
    
    def shuffle_data(self, data_dict):
        keys = list(data_dict.keys())
        random.shuffle(keys)  # Shuffle the keys
        shuffled_dict = {key: data_dict[key] for key in keys}
        return shuffled_dict
    def get_train_data(self):
        return self.train_data
    
    def get_validate_data(self):
        return self.validate_data
    def plot_samples(self, data_dict, title="Sample Images", num_samples=3):
            plt.figure(figsize=(10, num_samples * 3)) 
            for i, (key, (image, label)) in enumerate(list(data_dict.items())[:num_samples]):
                plt.subplot(num_samples, 1, i + 1)  
                plt.imshow(image.reshape(28, 28), cmap='gray')
                
                index = np.argmax(label)
                plt.title(f"Label: {label}\nIndex: {index}")
                
                plt.axis('off')
            plt.suptitle(title)
            plt.tight_layout()  
            plt.show()



class EMNISTLoader:
    def __init__(self, path, num_classes=62):
        self.path = path
        self.num_classes = num_classes

    def load_gz_file(self, filename, is_images=True):
        with gzip.open(self.path + filename, 'rb') as f:
            if is_images:
                f.read(16)  # Skip the header for images
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            else:
                f.read(8)  # Skip the header for labels
                data = np.frombuffer(f.read(), dtype=np.uint8)
            return data

    def load_data(self):
        train_images = self.load_gz_file('emnist-byclass-train-images-idx3-ubyte.gz', is_images=True)
        train_labels = self.load_gz_file('emnist-byclass-train-labels-idx1-ubyte.gz', is_images=False)  
        test_images = self.load_gz_file('emnist-byclass-test-images-idx3-ubyte.gz', is_images=True)
        test_labels = self.load_gz_file('emnist-byclass-test-labels-idx1-ubyte.gz', is_images=False) 
        
    
        train_labels = self.one_hot_encode(train_labels)
        test_labels = self.one_hot_encode(test_labels)

        train_data = {i: (train_images[i].flatten(), train_labels[i]) for i in range(len(train_images))}
        test_data = {i: (test_images[i].flatten(), test_labels[i]) for i in range(len(test_images))}
        
        return train_data, test_data

    def one_hot_encode(self, labels):
        one_hot_labels = np.zeros((labels.size, self.num_classes))
        one_hot_labels[np.arange(labels.size), labels] = 1
        return one_hot_labels

    def shuffle_data(self, data):
        keys = list(data.keys())
        np.random.shuffle(keys)
        shuffled_data = {key: data[key] for key in keys}
        return shuffled_data
    def plot_samples(self, data_dict, title="Sample Images", num_samples=3):
        plt.figure(figsize=(10, num_samples * 3)) 
        for i, (key, (image, label)) in enumerate(list(data_dict.items())[:num_samples]):
            plt.subplot(num_samples, 1, i + 1)  
            plt.imshow(image.reshape(28, 28), cmap='gray')
            
            index = np.argmax(label)
            plt.title(f"Label: {label}\nIndex: {index}")
            
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()  
        plt.show()
