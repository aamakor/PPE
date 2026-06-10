# This file contains the settings and run for the Preference Pareto Exploration (PPE) framework for multi-objective optimization in multi-task learning settings. 
# It loads the data, initializes the model and optimizers, and calls the main run function that implements the PPE algorithm. The results are saved in a pickle file for later analysis.
import argparse
import os
from pathlib import Path
# Third-party library imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, minres

# Local application imports
from Data.dataLoader import *
from Data.mnist_data import *
from Data.uci_data import *
from Data.uci_dataplus import *
from src.function import *
from src.models import MLP, MultiTaskNet56, MLP_uci_3plus
from src.util import *




def parse_arguments():
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Multi-Task/Objective Optimization Experiment.")
    # Dataset and Model Config
    parser.add_argument(
        "--dtype",
        type=str,
        default="UCI",
        choices=["UCI", "INT", "UCI_plus"],
        help="Dataset type: UCI, INT (MultiMNIST), or UCI_plus",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256, 
        help="Batch size for data loaders"
    )

    # Hyperparameters
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        help="Learning rate for primary optimizer"
    )
    parser.add_argument(
        "--num_obj", 
        type=int, 
        default=3, 
        help="Number of objectives"
    )
    parser.add_argument(
        "--num_init", 
        type=int, 
        default=500, 
        help="Number of initialization iterations"
    )
    parser.add_argument(
        "--num_pred", 
        type=int, 
        default=10, 
        help="Number of prediction steps"
    )
    parser.add_argument(
        "--num_corr", 
        type=int, 
        default=15, 
        help="Number of correction steps"
    )

    return parser.parse_args()


def main():
    # 1. Parse Inputs
    args = parse_arguments()

    # 2. Reproducibility Seeds
    np.random.seed(24)
    torch.manual_seed(24)

    # 3. Hardware Device Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    print("Current device:", device)

    # 4. Model Selection Dynamic Mapping
    if args.dtype == "UCI":
        model = MLP()
    elif args.dtype == "INT":
        model = MultiTaskNet56()
    elif args.dtype == "UCI_plus":
        model = MLP_uci_3plus()
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    # 5. Data Loading
    trainloader, valloader, testloader = Dataload(args.dtype, batch_size=args.batch_size)

    # 6. Safe System Path Resolution
    try:
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        FILE_DIR = Path(os.getcwd()).resolve()

    model_path = FILE_DIR.parent / f"PPE/model_path/{args.dtype}"
    model_path.mkdir(parents=True, exist_ok=True)

    # 7. Multi-GPU Setup & Device Placement
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Unwrap if DataParallel was used, matching your legacy setup
    model = model.module if isinstance(model, nn.DataParallel) else model

    # 8. Optimizers & Schedulers training and correction steps
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_init, eta_min=1e-5)

    optimizer_c = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    lr_sche_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=args.num_corr, eta_min=1e-5)

    # 9. Execution
    run(
        model=model,
        lr=args.lr,
        nobj=args.num_obj,
        optimizer=optimizer,
        optimizer_c=optimizer_c,
        num_init=args.num_init,
        path=model_path,
        lr_scheduler=lr_sche,
        lr_scheduler_c=lr_sche_c,
        criterion=criterion,
        num_pred=args.num_pred,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        device=device,
        type=args.dtype,
        num_corr=args.num_corr,
        num_minres=100,
    )


if __name__ == "__main__":
    main()
