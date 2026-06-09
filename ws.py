# This file implements the weighted sum baseline for multi-objective optimization. It trains a model using a weighted sum of the losses for each objective, 
# where the weights are given by the alpha vector corresponding to the preference from our PPE algorithm. The results are saved in a pickle file for later analysis.

import torch
import torch.nn as nn
from pathlib import Path
import os
import pickle
from Data.dataLoader import *
from Data.uci_data import *
from Data.uci_dataplus import *
from src.models import *
from src.function import *
from  src.util import *
from scipy.sparse.linalg import LinearOperator, minres
from torch.nn.utils import parameters_to_vector, vector_to_parameters   
import numpy as np


def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x

def run_ws(idx,preference, model, optimizer, trainloader,valloader,testloader,scheduler, device, criterion, alpha, type, num_epochs=100):

    torch.cuda.synchronize()
    total_start = time.time()
    for epoch in range(num_epochs):
        # ---- TRAIN ----
        torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_samples = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(images)

            losses = [
                criterion(outs[i], labels[:, i])
                for i in range(len(outs))
            ]

            # ---- WEIGHTED SUM OBJECTIVE ----
            alpha_t = torch.tensor(alpha, dtype=losses[0].dtype, device=losses[0].device)
            #weighted_loss = sum(alpha_t[i] * losses[i] for i in range(len(losses)))
            weighted_loss = torch.dot(alpha_t, torch.stack(losses))

            weighted_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += weighted_loss.item() * images.size(0)
            num_samples += images.size(0)

        train_loss = total_loss / num_samples
        torch.cuda.synchronize()
        train_time = time.time() - t0
        
        # ---- VALIDATION ----

        model.eval()
        val_losses = torch.zeros(len(alpha), device=device)
        num_val = 0

        with torch.no_grad():
            for images, labels in valloader:
                images = images.to(device)
                labels = labels.to(device)

                outs = model(images)

                for i in range(len(alpha)):
                    val_losses[i] += (
                        criterion(outs[i], labels[:, i]).item()
                        * images.size(0)
                    )

                num_val += images.size(0)

        val_losses /= num_val
        #val_loss_scalar = torch.dot(alpha, val_losses).item()

        #scheduler.step(val_loss_scalar)

        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val losses: {val_losses.tolist()}")
    
    torch.cuda.synchronize()
    total_time = time.time() - total_start
    # Final evaluation on test set
    model.eval()
    test_losses = torch.zeros(len(alpha), device=device)
    num_test = 0    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outs = model(images)

            for i in range(len(alpha)):
                test_losses[i] += (
                    criterion(outs[i], labels[:, i]).item()
                    * images.size(0)
                )

            num_test += images.size(0)
    test_losses /= num_test
    
    print(f"alpha: {alpha.tolist()}, val_losses: {val_losses.tolist()}")
    print(f"Test losses: {test_losses.tolist()}")
    print(f"Total training time: {total_time:.2f} seconds")

    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    file_path = FILE_DIR.parent / f"PPE/Results/{type}_ws_{num_epochs}"
    file_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists

    if preference is None:

        with open(file_path / f'first_result_ws{idx}.pkl', 'wb') as f:
                pickle.dump((np.array([1/3]*len(alpha)),
                        to_cpu(alpha),
                        to_cpu(val_losses),
                        to_cpu(test_losses),
                    ),
                    f
                )
    else:
        with open(file_path / f'first_result_ws{idx}.pkl', 'wb') as f:
            pickle.dump((to_cpu(preference),
                    to_cpu(alpha),
                    to_cpu(val_losses),
                    to_cpu(test_losses),
                ),
                f
            )

        

    np.save(file_path / f"info_ws{idx}.npy", {"total_time": total_time})
    return test_losses.tolist(), total_time


if __name__ == "__main__":
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')  # use default cuda device
        import torch.backends.cudnn as cudnn  # make cuda deterministic
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        device = torch.device('cpu') # otherwise use cpu

    print('Current device:', device)


    dtype = "UCI" #"INT"  #   "UCI_plus"  # 
    """
    # Uncomment the model you want to use. 
    * MLP () for UCI 3-task problem
    * while MultiTaskNet56 is the model rchitecture designed for the MultiMNIST dataset (INT).
    #  MLP_uci_3plus is a specific architecture for the  UCI 5-task  (UCI_plus) dataset.

    """
    model = MLP().to(device) #MultiTaskNet56() #  MLP_uci_3plus().to(device) 
    batch_size = 256
    num_epochs = 10 #50#100 # 500 # (2  is too poor)
    lr =  0.01 
    n = 6 #number of preferences to run ws baseline on; should match number of preferences from CM runs

    trainloader, valloader, testloader= Dataload(dtype,batch_size=batch_size) #, mr_dataloader 

    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    model_path = FILE_DIR.parent / f"PPE/Results/{dtype}_1&3"
    file =  FILE_DIR.parent / f"PPE/Results/{dtype}_ws_{num_epochs}"
    file.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.module if isinstance(model, nn.DataParallel) else model #code sometimes uses single-GPU (not wrapped) or multi-GPU
    optimizer =  torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    #torch.optim.Adam(model.parameters(), lr=1e-3)  #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=5e-4)
    lr_sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-5)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30 * len(trainloader))

    #with open(file / f'info_0ws.txt', "a") as f:
    #Initial optimal point
    alpha_init = np.load(model_path  /f"alpha_500.npy")
    _, total_time = run_ws(0, None, model, optimizer, trainloader,valloader,testloader,lr_sche, device, criterion, alpha_init,dtype, num_epochs=num_epochs)
    #    f.write("=== Training ===\n")
    #    f.write(f"Alpha {alpha_init}\n")
    #    f.write(f"Training time (total)0: {total_time:.2f} s\n \n")

    
    for j in range(n):
        file = model_path / f'first_alphas_cen{j}.pkl'

        if not os.path.exists(file):
                # Skip if the file does not exist
                continue
        with open(file, 'rb') as f:
            preference, alpha = pickle.load(f)
    
        alpha = alpha[1]
        preference = preference[0]

        #*****************RUN WEIGHTED SUM BASELINE ***********************#
        print(f"Running weighted sum baseline for preference {preference} with alpha {alpha}")
        _, total_time = run_ws(j+1,preference, model, optimizer, trainloader,valloader,testloader,lr_sche, device, criterion, alpha,dtype, num_epochs=num_epochs)


                
        



    