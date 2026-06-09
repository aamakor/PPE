from .mnist_data import *
from .uci_data import *
from .uci_dataplus import *

def Dataload(type= "UCI", batch_size = 256):
    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    DATA_DIR = FILE_DIR.parent / f'Data/{type}' 

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if type == "UCI":
        trainset = UCI(DATA_DIR, train=True, download=False)
        testset = UCI(DATA_DIR, train=False, download=False)

    elif type == "UCI_plus":
        trainset = UCI_plus(DATA_DIR, train=True, download=False) # Set to True if data not downloaded
        testset = UCI_plus(DATA_DIR, train=False, download=False)

    elif type == "INT":
        trainset, testset = MergedDataset(merged_images=True).Create_mergedDataset()

    else:
        raise ValueError("Invalid dataset type. Choose from 'UCI', 'UCI_plus', or 'INT' or add a new dataset class in data_func.py")

        
    # Split the trainset into training and validation sets
    train_size = int(0.8 * len(trainset))  # 80% for training
    val_size = len(trainset) - train_size   # 20% for validation
    train_set, val_set = random_split(trainset, [train_size, val_size],  generator=torch.Generator().manual_seed(24))

 
    # Create DataLoaders
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=0)
   


    return trainloader, valloader, testloader