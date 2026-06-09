# This file contains the settings and run for the Preference Pareto Exploration (PPE) framework for multi-objective optimization in multi-task learning settings. 
# It loads the data, initializes the model and optimizers, and calls the main run function that implements the PPE algorithm. The results are saved in a pickle file for later analysis.
from Data.mnist_data import *
from Data.dataLoader import *
from Data.uci_data import *
from Data.uci_dataplus import *
from src.models import *
from src.function import *
from  src.util import *
from scipy.sparse.linalg import LinearOperator, minres
from torch.nn.utils import parameters_to_vector, vector_to_parameters   
np.random.seed(24)
torch.manual_seed(24)



# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')  # use default cuda device
    import torch.backends.cudnn as cudnn  # make cuda deterministic
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    device = torch.device('cpu') # otherwise use cpu

print('Current device:', device)


dtype =  "UCI"#"INT"  #"UCI_plus"  #   Uncomment to run on UCI datasets(UCI), or MultiMNIST (INT) or UCI_plus (UCI with more tasks)
model =  MLP() # MultiTaskNet56()  #MLP_uci_3plus() #    #   Uncomment to run on UCI model (MLP()), or MultiMNIST (MultiTaskNet56()) or UCI_plus (UCI with more tasks- MLP_uci_3plus())

num_obj = 3 # 5 #number of objectives
num_init = 500 #500  #500
num_pred = 10  #30
num_corr = 15 #0
batch_size = 256

trainloader, valloader, testloader= Dataload(dtype,batch_size=batch_size) #, mr_dataloader 

# Load the MNIST dataset'
try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()
model_path = FILE_DIR.parent / f"PPE/model_path/{dtype}"
model_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists
#pref,lr = [np.array([0, -0.7,-0.3])], 0.01, #0.001 for adam
lr =  0.01 #0.01 #

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
model = model.module if isinstance(model, nn.DataParallel) else model #code sometimes uses single-GPU (not wrapped) or multi-GPU
optimizer =  torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4) 
#torch.optim.Adam(model.parameters(), lr=1e-3)  #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=5e-4)

#sch_heads  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_heads, T_max=10, eta_min=1e-5)
lr_sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_init,eta_min=1e-5)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30 * len(trainloader))

optimizer_c = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-5) #torch.optim.Adam(model.parameters(), lr=1e-3) # torch.optim.SGD(model.parameters(), lr=0.005)# , weight_decay=5e-4, nesterov=True) #  ##,  
lr_sche_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c,T_max=num_corr,eta_min=1e-5)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, 30 * len(trainloader)) # #CosineAnnealingLR(optimizer_c, 30 * len(trainloader))

run(model,lr,num_obj, optimizer,optimizer_c, num_init, model_path,lr_sche,lr_sche_c,criterion,  num_pred,trainloader,valloader, testloader, device, type = dtype,num_corr= num_corr, num_minres= 100)#mr_dataloader, num_minres=500)



