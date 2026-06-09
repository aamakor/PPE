#### Data loading and preprocessing for MultiMNINST dataset (FashionMNIST, KMNIST and MNIST merged together)

import pickle
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision import transforms
import matplotlib
matplotlib.use('webAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from pathlib import Path
from torch.utils.data import RandomSampler
import os
SEED = 24   # or any fixed number you like
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)   # if using GPU
torch.backends.cudnn.deterministic = True   # optional: more reproducibility

def merge_3_image(first_image, second_image, third_image):
    # Convert the images to PIL format
    first_image = Image.fromarray(np.array(first_image))
    second_image = Image.fromarray(np.array(second_image))
    third_image = Image.fromarray(np.array(third_image))
    # Create a new RGB blanck image
    merged_image = Image.new('L', (56, 56))

    positions = [(0, 28), (28, 0), (0, 0), (28, 28)]

    # Shuffle the positions randomly
    random.shuffle(positions)
    # Paste the images into the new image at the shuffled positions
    merged_image.paste(first_image, positions[0])
    merged_image.paste(second_image, positions[1])
    merged_image.paste(third_image, positions[2])
    # Convert the padded image to a numpy array and return it
    merged_image = np.array(merged_image)
    return merged_image

# Merging two datasets together
def merging_3(first_dataset, second_dataset, third_dataset):
    # Set the random seed for reproducibility
    np.random.seed(123)
    merged_images = []
    N = len(first_dataset)
    for i in range(N):
        # Get the images and labels from both datasets
        first_image, first_label = first_dataset[i]
        second_image, second_label = second_dataset[i]
        third_image, third_label = third_dataset[i]
        merged_image = merge_3_image(first_image, second_image, third_image)
        # The label of the new image is a couple of the three labels
        merged_images.append((merged_image, (first_label, second_label, third_label)))

    return merged_images, first_dataset.classes
class MergedDataset(torch.utils.data.Dataset):

    def __init__(self, merged_images= None, transform=None):
        self.transform = transform
        self.merged_images = merged_images

            
    def __getitem__(self, index):
        image, (label1, label2, label3 ) = self.merged_images[index]
        label = torch.tensor([label1, label2, label3])
        #image, label = self.merged_images[index]
        if self.transform:
            image = self.transform(image)
        return image, label #return image, (label1, label2, label3 ) 

    def __len__(self):
        return len(self.merged_images)
    

    def download(self):
        if self._check_exists():
            return

    def Create_mergedDataset(self):

        # Load the MNIST dataset'
        try:
            # Works in normal Python scripts
            FILE_DIR = Path(__file__).resolve().parent
        except NameError:
            # Fallback for Jupyter/IPython
            FILE_DIR = Path(os.getcwd()).resolve()
        DATA_DIR = FILE_DIR.parent / "Data/VarMNIST"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR1 = FILE_DIR.parent / "Data/mergeMnist"
        DATA_DIR1.mkdir(parents=True, exist_ok=True)

        def check_exists():
            return (DATA_DIR1 /'train_FKMnist_merged_images.pkl').is_file() and \
                (DATA_DIR1 /'test_FKMnist_merged_images.pkl').is_file() and \
                (DATA_DIR1 /'train_set.pkl').is_file() and \
                (DATA_DIR1 /'test_set.pkl').is_file()
        
        if check_exists():
            #print('Loading merged images from files...')
            with open(DATA_DIR1 /'train_FKMnist_merged_images.pkl', 'rb') as f:
                train_FKMnist_merged_images = pickle.load(f)
            with open(DATA_DIR1 /'test_FKMnist_merged_images.pkl', 'rb') as f:
                test_FKMnist_merged_images = pickle.load(f)

            with open(DATA_DIR1 /'train_set.pkl', 'rb') as f:
                trainset = pickle.load(f)
            with open(DATA_DIR1 /'test_set.pkl', 'rb') as f:
                testset = pickle.load(f)
            
            with open(DATA_DIR1 /'classes.pkl', 'rb') as f:
                classes = pickle.load(f)
            
            #return train_FKMnist_merged_images, test_FKMnist_merged_images, classes
            return trainset, testset
        
        
        train_mnist_dataset = MNIST(root=DATA_DIR, train=True, download= True)
        test_mnist_dataset = MNIST(root=DATA_DIR, train=False, download= True)
        train_fmnist_dataset = FashionMNIST(root=DATA_DIR, train=True, download= True)
        test_fmnist_dataset = FashionMNIST(root=DATA_DIR, train=False, download= True)
        train_kmnist_dataset = KMNIST(root=DATA_DIR, train=True, download= True)
        test_kmnist_dataset = KMNIST(root=DATA_DIR, train=False, download= True)

    

        # Merge train for 3
        train_FKMnist_merged_images, classes = merging_3(train_fmnist_dataset,  train_kmnist_dataset, train_mnist_dataset)
        # Merge test for 3
        test_FKMnist_merged_images, classes = merging_3(test_fmnist_dataset,test_kmnist_dataset, test_mnist_dataset)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ## Uncomment if you want to extract small data set
        #train_subset = extract_subset(train_subset, 500) #
        #test_subset = extract_subset(test_subset, 50)

        #To transform data into a custom class
        trainset = MergedDataset(merged_images=train_FKMnist_merged_images, transform=transform)
        testset = MergedDataset(merged_images=test_FKMnist_merged_images, transform=transform)
        # Save the merged images to a file
        with open(DATA_DIR1 /'train_FKMnist_merged_images.pkl', 'wb') as f:
            pickle.dump(train_FKMnist_merged_images, f)
        with open(DATA_DIR1 /'test_FKMnist_merged_images.pkl', 'wb') as f:
            pickle.dump(test_FKMnist_merged_images, f)
        
        with open(DATA_DIR1 /'train_set.pkl', 'wb') as f:
            pickle.dump(trainset, f)
        with open(DATA_DIR1 /'test_set.pkl', 'wb') as f:
            pickle.dump(testset, f)

        with open(DATA_DIR1 /'classes.pkl', 'wb') as f:
            pickle.dump(classes, f)
        
        #return train_FKMnist_merged_images, test_FKMnist_merged_images, classes
        return trainset, testset


# Define a custom dataset class for the merged images


#train_FKMnist_merged_images, test_FKMnist_merged_images,classes = MergedDataset(merged_images=True).Create_mergedDataset()
# Visualize an example
def plot_Imgdata3(merged_images,classes):
    
    images, labels =  zip(*merged_images[:20])

    # Create a figure with subplots for each image
    fig, axes = plt.subplots(nrows=4, ncols=5)

    # Loop through the images and labels and display them in the subplots
    for i, (image, label) in enumerate(zip(images, labels)):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        ax.imshow(image, cmap= "gray")
        ax.set_title(f'{classes[label[0]], label[1], label[2]}', fontsize=6)
        ax.axis('off')
    plt.tight_layout()
    #plt.savefig("Data/Mnist/Three10_images.png")
    plt.show()
    
    
#plot_Imgdata3(train_FKMnist_merged_images,classes)



##### In order to test our algorithm for DNN we first extract a small subset of the dataset and then we will use a small neural network to fit this small dataset. 
##### The neural network will have two loss functions, one for each digit in the image.
def extract_subset(full_dataset, subset_size):
    """
    Extracts a random subset from a full dataset.
    
    Args:
        full_dataset (Dataset): The original dataset (e.g., train or test set).
        subset_size (int): The desired size of the subset.

    Returns:
        Subset: A new dataset representing the random subset.
    """
    if subset_size > len(full_dataset):
        raise ValueError("Subset size cannot be larger than the full dataset size.")
    
    # Generate random indices for the subset
    indices = torch.randperm(len(full_dataset))[:subset_size]
    
    # Create a Subset object from the generated indices
    return Subset(full_dataset, indices)



def get_random_batch(trainloader):
    dataset = trainloader.dataset
    batch_size = trainloader.batch_size
    sampler = RandomSampler(dataset)
    new_trainloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return next(iter(new_trainloader))

