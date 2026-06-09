### Pareto MTL implementation for multi-objective multi-task learning (MOO-MTL) problems.
#The results are saved in a pickle file for later analysis.

import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable

from model_reg import RegressionTrain


from min_norm_solvers import MinNormSolver

import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

#from PPE.Data.data_func import *
from Data.dataLoader import *
from Data.uci_data import *
from Data.uci_dataplus import *
from src.models import *
#from PPE.src.function import *
#from  PPE.src.util import *
from scipy.sparse.linalg import LinearOperator, minres
from torch.nn.utils import parameters_to_vector, vector_to_parameters   
np.random.seed(24)
torch.manual_seed(24)
import tqdm
import time


def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x

def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight2 =  torch.sum(torch.stack([sol[j] * w[idx][j ,2] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1,weight2])
   
    
    return flag, weight


def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight2 =  sol[2] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,2] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1,weight2])
    
    return weight


def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def circle_points(r_list, n_list):
    """
    Generate evenly distributed unit preference vectors for three tasks
    (points on positive octant of a sphere).
    """
    spheres = []

    for r, n in zip(r_list, n_list):
        theta = np.linspace(0, 0.5*np.pi, n)
        phi = np.linspace(0, 0.5*np.pi, n)

        theta, phi = np.meshgrid(theta, phi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        spheres.append(points)

    return spheres

def train(dtype, train_loader,val_loader,test_loader,model, niter, optimizer,scheduler,ref_vec, prefs =None, n_tasks= 3, pref_idx = 0):


    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []
    
    # print the current preference vector
    print(f'Preference Vector ({prefs[pref_idx]}/ optimal weights: {ref_vec[pref_idx].cpu().numpy()}')
   

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    torch.cuda.synchronize()
    total_start = time.time()
    for t in range(2):
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []
            
            
            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
                
                grads[i] = []
                
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
            
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
            #print(weight_vec)
            
            # early stop once a feasible solution is obtained
            if flag == True:
                print("feasible solution is obtained.")
                break
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break
                
        

    # run niter epochs of ParetoMTL 
    for t in tqdm.trange(niter):
        
        scheduler.step()
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # obtain and store the gradient 
            grads = {}
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
            
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment              
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
                
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
            
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()


        # calculate and record performance
        #if t == 0 or (t + 1) % 2 == 0:
        # Validate the model on the validation set    
        model.eval()
        with torch.no_grad():

            total_train_loss = []
            train_acc = []
    
            correct1_train = 0
            correct2_train = 0
            correct3_train = 0
            
            
            for (it, batch) in enumerate(val_loader):
                
                X = batch[0]
                ts = batch[1]
                if torch.cuda.is_available():
                    X = X.cuda()
                    ts = ts.cuda()
    
                valid_train_loss = model(X, ts)
                total_train_loss.append(valid_train_loss)
                output1 = model.model(X)[0].max(1, keepdim=True)[1]#[:,0]
                output2 = model.model(X)[1].max(1, keepdim=True)[1]#[:,1]
                output3 = model.model(X)[2].max(1, keepdim=True)[1]#[:,2]
                correct1_train += output1.eq(ts[:,0].view_as(output1)).sum().item()
                correct2_train += output2.eq(ts[:,1].view_as(output2)).sum().item()
                correct3_train += output3.eq(ts[:,2].view_as(output3)).sum().item() 


                
                
            train_acc = np.stack([1.0 * correct1_train / len(val_loader.dataset),1.0 * correct2_train / len(val_loader.dataset),1.0 * correct3_train / len(val_loader.dataset)])
    
            total_train_loss = torch.stack(total_train_loss)
            average_train_loss = torch.mean(total_train_loss, dim = 0)
            
        
        # record and print
        if torch.cuda.is_available():
            
            task_train_losses.append(average_train_loss.data.cpu().numpy())
            train_accs.append(train_acc)
            
            weights.append(weight_vec.cpu().numpy())
            
            print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                    t + 1, niter,  weights[-1], task_train_losses[-1],train_accs[-1]))                 
               
    torch.cuda.synchronize()
    total_time = time.time() - total_start
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():

        total_test_loss = []
        test_acc = []

        correct1_test = 0
        correct2_test = 0
        correct3_test = 0


        for (it, batch) in enumerate(test_loader):
            
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            valid_test_loss = model(X, ts)
            total_test_loss.append(valid_test_loss)
            output1 = model.model(X)[0].max(1, keepdim=True)[1]#[:,0]
            output2 = model.model(X)[1].max(1, keepdim=True)[1]#[:,1]
            output3 = model.model(X)[2].max(1, keepdim=True)[1]#[:,2]
            correct1_test += output1.eq(ts[:,0].view_as(output1)).sum().item()
            correct2_test += output2.eq(ts[:,1].view_as(output2)).sum().item()
            correct3_train += output3.eq(ts[:,2].view_as(output3)).sum().item() 


            
            
        test_acc = np.stack([1.0 * correct1_test / len(test_loader.dataset),1.0 * correct2_test / len(test_loader.dataset),1.0 * correct3_test / len(test_loader.dataset)])

        total_test_loss = torch.stack(total_test_loss)
        average_test_loss = torch.mean(total_test_loss, dim = 0)

    print(f"alpha: {ref_vec.tolist()}, val_losses: {average_train_loss.tolist()}")
    print(f"Test losses: {average_test_loss.tolist()}")
    print(f"Total training time: {total_time:.2f} seconds")

    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    file_path = FILE_DIR.parent.parent / f"PPE/Results/{dtype}_pareto_mtl_{niter}"
    file_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists

    if prefs[pref_idx] is None:

        with open(file_path / f'first_result_pareto_mtl{pref_idx}.pkl', 'wb') as f:
                pickle.dump((np.array([1/3]*len(ref_vec[pref_idx])),
                        to_cpu(ref_vec[pref_idx]),
                        to_cpu(average_train_loss),
                        to_cpu(average_test_loss),
                    ),
                    f
                )
    else:
        with open(file_path / f'first_result_pareto_mtl{pref_idx}.pkl', 'wb') as f:
            pickle.dump((to_cpu(prefs[pref_idx]),
                    to_cpu(ref_vec[pref_idx]),
                    to_cpu(average_train_loss),
                    to_cpu(average_test_loss),
                ),
                f
            )

        


    np.save(file_path / f"info_pareto_mtl{pref_idx}.npy", {"total_time": total_time})
    #return test_losses.tolist(), total_time



def run( niter = 100):
    """
    run Pareto MTL
    """
    
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')  # use default cuda device
        import torch.backends.cudnn as cudnn  # make cuda deterministic
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        device = torch.device('cpu') # otherwise use cpu

    print('Current device:', device)
    
    dtype =  "UCI"
    batch_size = 256
    trainloader, valloader, testloader= Dataload(dtype,batch_size=batch_size)

    n = 6
    batch_size = 256
  
    print('==>>> total trainning batch number: {}'.format(len(trainloader)))
    print('==>>> total validation batch number: {}'.format(len(valloader)))
    print('==>>> total testing batch number: {}'.format(len(testloader))) 
    
    
    # define the base model for ParetoMTL  

    model = RegressionTrain(MLP())

    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    model_path = FILE_DIR.parent.parent / f"PPE/Results/{dtype}_1&3"

    if torch.cuda.is_available():
        model.cuda()

    model.to(device)
    lr = 0.01

    optimizer =  torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=niter,eta_min=1e-5) #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)
    
    # generate #npref preference vectors      
    n_tasks = 3

    ref_vec = []
    preferences = []

    alpha_init = np.load(model_path  /f"alpha_500.npy") #{niter}
    #alpha_init = np.array(alpha_init) / np.linalg.norm(np.array(alpha_init))
    ref_vec.append(alpha_init)
    preferences.append(None)
    #alpha_init = torch.from_numpy(alpha_init).float().cuda()
    #train(dtype ,trainloader, valloader,testloader, model, niter,optimizer, scheduler, alpha_init,None, n_tasks)
    
    for j in range(n):
        file = model_path / f'first_alphas_cen{j}.pkl'

        if not os.path.exists(file):
                # Skip if the file does not exist
                continue
        with open(file, 'rb') as f:
            preference, alpha = pickle.load(f)
    
        #alpha = alpha[1]
        #alpha = torch.from_numpy(alpha).float().cuda()
        #preference = preference[0]
        alpha = alpha[1] #np.array(alpha[1]) / np.linalg.norm(np.array(alpha[1]))
        ref_vec.append(alpha)
        preferences.append(preference[0])
    #ref_vec = torch.tensor(ref_vec).cuda().float()
    ref_vec = torch.from_numpy(np.array(ref_vec)).cuda().float()
    #print(ref_vec)

    for i in range(n+1):
        #if i >= 5:
        pref_idx = i 
        train(dtype, trainloader, valloader, testloader, model, niter, optimizer, scheduler, ref_vec, preferences, n_tasks, pref_idx)
        
run(niter = 50)




