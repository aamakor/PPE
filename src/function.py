
from itertools import combinations
from pathlib import Path
import os
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, minres
from scipy.optimize import nnls
#from torch.autograd import Variable
from cvxopt import matrix, solvers
np.random.seed(24)
torch.manual_seed(24)
np.set_printoptions(precision=8)
#import tqdm as tq
from tqdm import tqdm, trange
import time
from contextlib import contextmanager
import shutil
from Data.mnist_data import get_random_batch
from Data.dataLoader import *
from scipy.optimize import nnls


# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')  # use default cuda device
    import torch.backends.cudnn as cudnn  # make cuda deterministic
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    device = torch.device('cpu') # otherwise use cpu

#print('Current device:', device)


def _min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)
    # v2v2 - gamm * gamma * (v1 - v2)^2
    # cost = v2v2 - gamma * gamma * (v1v1 + v2v2 - 2 * v1v2)
    #      = v2v2 - gamma * (v2v2 - v1v2)
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def _min_norm_2d(vecs):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = None
    dps = vecs.matmul(vecs.t()).cpu().numpy()
    for i, j in combinations(range(len(vecs)), 2):
        c, d = _min_norm_element_from2(dps[i, i], dps[i, j], dps[j, j])
        if dmin is None:
            dmin = d
        if d <= dmin:
            dmin = d
            sol = [(i, j), c, d]
    return sol, dps


def _projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return np.maximum(y - tmax_f, np.zeros(y.shape))


def _next_point(cur_val, grad, n):
    proj_grad = grad - (np.sum(grad) / n)
    tm1 = -cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    t = 1
    if len(tm1[tm1 > 1e-7]) > 0:
        t = np.min(tm1[tm1 > 1e-7])
    if len(tm2[tm2 > 1e-7]) > 0:
        t = min(t, np.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point


def find_min_norm_element(vecs, max_iter=250, stop_crit=1e-5):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
    the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    # Solution lying at the combination of two points
    init_sol, dps = _min_norm_2d(vecs.detach())

    n = len(vecs)
    sol_vec = np.zeros(n)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec, init_sol[2]

    iter_count = 0

    while iter_count < max_iter:
        grad_dir = -1.0 * np.dot(dps, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)
        # Re-compute the inner products for line search
        v1v1 = 0.0
        v1v2 = 0.0
        v2v2 = 0.0
        for i in range(n):
            for j in range(n):
                v1v1 += sol_vec[i] * sol_vec[j] * dps[i, j]
                v1v2 += sol_vec[i] * new_point[j] * dps[i, j]
                v2v2 += new_point[i] * new_point[j] * dps[i, j]
        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc * sol_vec + (1 - nc) * new_point
        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < stop_crit:
            break
        sol_vec = new_sol_vec
    return sol_vec, nd


def compute_alpha(jacobians):
    sol, min_norm = find_min_norm_element(jacobians)
    return sol



def flatten_grads(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])

def task_grads_shared(loss_t, shared_params_list):
    grads = torch.autograd.grad(loss_t, shared_params_list, retain_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params_list)]
    return flatten_grads(grads)  # [n_shared]

def mgda_optimize(model,images,labels,criterion,compute_alpha,optimizer,lr_scheduler = None):
    """
    MGDA optimization with safe backtracking line search and
    integrated gradient averaging (microbatches + EMA).

    ema_state must be a dict passed by the caller to persist smoothing.
    """

    m = labels.size(1)
    shared_params = [p for p in model.parameters() if p.requires_grad]
    # ---- Forward: compute per-task losses
    outs = model(images)
    losses = [criterion(outs[i], labels[:, i].long()) for i in range(m)]

    G_rows = []

    with torch.enable_grad():
        for i in range(m):
            gi = task_grads_shared(losses[i], shared_params)
            G_rows.append(gi)

        G = torch.stack(G_rows, dim=0)

    # ---- Compute MGDA coefficients α on the simplex
    alpha = compute_alpha(G)
    # ---- Standard weighted-loss update
    #optimizer.zero_grad(set_to_none=True)
    weighted_loss = sum(float(alpha[i]) * losses[i] for i in range(m))
    weighted_loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return alpha


def topk_accuracies(logits, targets, ks=(1,)):
    assert logits.dim() == 2
    assert targets.dim() == 1
    assert logits.size(0) == targets.size(0)

    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    targets = targets.unsqueeze(1).expand_as(pred)
    correct = pred.eq(targets).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1).mean()
        accu_list.append(accu.item())
    return accu_list

def evaluate(model, testloader,m, criterion):
    num_samples = 0
    total_losses = np.zeros(m)
    total_top1s = np.zeros(m)
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            batch_size = len(images)
            num_samples += batch_size
            images = images.to(device)
            labels = labels.to(device)
            outs = model(images) 
            losses = [criterion(outs[i], labels[:, i].long()).item() for i in range(m)]
            total_losses += batch_size * np.array(losses)
            topks = [topk_accuracies(outs[i], labels[:, i].long())[0] for i in range(m)]
            total_top1s += batch_size * np.array(topks)
    total_losses /= num_samples
    total_top1s /= num_samples
    return total_losses, total_top1s


def compute_grads( model, criterion,trainloader, jacobian_trainiter,ratio=1.0):
    #global jacobian_trainiter
    num_batches = int(len(trainloader) * ratio)
    jacobians = None
    for _ in range(num_batches):
        try:
            images, labels = next( jacobian_trainiter)
        except StopIteration:
            jacobian_trainiter = iter(trainloader)
            images, labels = next(jacobian_trainiter)
        images = images.to(device)
        labels = labels.to(device)
        outs = model(images)
        losses = [criterion(outs[i], labels[:, i].long()) for i in range(labels.size(1))]
        param_grads = [list(torch.autograd.grad(
            l, model.parameters(), allow_unused=True,
            retain_graph=True, create_graph=False)) for l in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, model.parameters())):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
        sub_jacobians = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)
        sub_jacobians.detach_()
        if jacobians is None:
            jacobians = sub_jacobians
        else:
            jacobians.add_(sub_jacobians)
    jacobians.div_(num_batches)
    return jacobians.clone().detach()



def row_normalize_G(G, eps=1e-12):
    """Normalize each row of G; return normalized G and row norms."""
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    Gn = G / (norms + eps)
    zero_rows = (norms.squeeze(-1) < eps)
    Gn[zero_rows] = 0.0
    return Gn#, norms.squeeze(-1)


class HVPLinearOperator(LinearOperator):
    def __init__(self, dataloader,model, criterion):
        model_size = sum(p.numel() for p in model.parameters())
        shape = (model_size, model_size)
        dtype = list(model.parameters())[0].detach().cpu().numpy().dtype

        super(HVPLinearOperator, self).__init__(dtype, shape)

        self.dataloader = dataloader
        self.dataiter = iter(dataloader)
        self.model = model
        self.damping = 0.1
        self.criterion = criterion

        self.alpha_jacobians = None

    def _get_jacobians(self):
        try:
            images, labels = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            images, labels = next(self.dataiter)
        images = images.to(device)
        labels = labels.to(device)
        outs = self.model(images)
        losses = [self.criterion(outs[i], labels[:, i].long()) for i in range(labels.size(1))]
        param_grads = [list(torch.autograd.grad(
            l, self.model.parameters(), allow_unused=True,
            retain_graph=True, create_graph=True)) for l in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, self.model.parameters())):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
                    
        return torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)

    @contextmanager
    def init(self, alpha):
        try:
            alpha = torch.as_tensor(alpha.astype(self.dtype), device=device).view(1, -1)
            jacobians = self._get_jacobians()
            self.alpha_jacobians = alpha.matmul(jacobians).squeeze()
            yield self
        finally:
            self.alpha_jacobians = None
    def _matvec_tensor(self, tensor):
        dot = self.alpha_jacobians.dot(tensor)
        param_alphas_hvps = torch.autograd.grad(dot, self.model.parameters(), retain_graph=True)
        alphas_hvps = parameters_to_vector([p.contiguous() for p in param_alphas_hvps])

        if self.damping > 0.0:
            alphas_hvps.add_(tensor, alpha=self.damping)
        return alphas_hvps
    def _matvec(self, x):
        tensor = torch.as_tensor(x.astype(self.dtype), device=device)
        ret = self._matvec_tensor(tensor)        
        #print("tensor", tensor)
        return ret.detach().cpu().numpy()

def beta_basis(m=3):
    """Construct n x k matrix with row-sum 0 and full row rank (works if n <= k-1)."""
    k = m-1
    if k > m:
        raise ValueError("Impossible: need n <= k-1 for full row rank with row-sum 0.")
    B = np.zeros((k, m), dtype=float)
    for i in range(k):
        B[i, i] = 1.0
        B[i, i+1] = -1.0
    return B



def solve_theta_nnls(G, g_i):
    """
    Solve: min_{theta >= 0} 0.5||G theta + g_i||^2  (NNLS with A=G, b=-g_i)
    G: (n, m_sel), g_i: (n,)
    """
    A = np.asfortranarray(np.asarray(G, dtype=np.float64))  # NNLS likes float64 (Fortran contiguous ok)
    b = -np.asarray(g_i, dtype=np.float64) #-
    theta, _ = nnls(A, b)  # returns nonnegative solution
    return theta


def proj_general(V, d, device=None):
    V = V if torch.is_tensor(V) else torch.tensor(V)
    d = d if torch.is_tensor(d) else torch.tensor(d) 
    if device is None:
        device = V.device
    # ensure correct shapes
    V = V.float().T               # (k,n)
    d = d.float().ravel()         # (n,)

    M = V.T @ V                   # (k,k)

    I = torch.eye(M.shape[0], device=device)

    if torch.allclose(M, I):
        return V @ (V.T @ d)      # (n,)
    return V @ torch.linalg.solve(M, V.T @ d)

def check_combinations(arr,max_index, checkTwo = False, index2 = None):
    if len(max_index) >1:
        print(len(max_index), "max indices:", max_index)
        for i in range(len(max_index)):
            if arr[max_index[0]] < 0 and arr[i] == 0:
                return arr
            else:
                print(f"❌ change preference must exclude objective {max_index[0]+1}")
    if arr[max_index] < 0:
        print(f"✅ Valid preference {arr}")
        return arr
    else:
        print(f"❌ change preference must include objective {max_index+1}")
    
    """if checkTwo == True:
        if len(index2) > 1 :
            for i in index2:
                if arr[max_index] < 0 and arr[i] == 0:
                    return arr
                else:
                    print(f"❌ change preference must exclude objective {index2+1}")"""
    return None



   

def backtracking_step(model, criterion, trainloader, v, grad, eta_init=1.0, beta=0.5, c1=1e-4, pref=None):
    """
    Backtracking line search with preference conditions for PyTorch neural networks.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network to update.
    loss_fn : callable
        Loss function taking (pred, target) -> scalar tensor.
    trainloader : Dataset
        Data used to evaluate loss.
    v : torch.Tensor
        Search direction (usually -grad).
    grad : list of torch.Tensor
        Gradients for each objective.
    eta_init : float
        Initial step size (default 1.0).
    beta : float
        Reduction factor for eta (default 0.5).
    c1 : float
        Armijo constant (default 1e-4).
    pref : list or np.ndarray
        Preference vector indicating which objectives to consider.

    Returns
    -------
    eta : float
        Step size satisfying Armijo condition (0 if none found).
    """
    # Get a fixed batch for backtracking
    images, labels = get_random_batch(trainloader) #next(iter(trainloader))  # Use the first batch
    images = images.to(device)
    labels = labels.to(device)

    # Compute initial loss (f) on the fixed batch
    with torch.no_grad():
        outs = model(images)
        f = [criterion(outs[i], labels[:, i].long()).item() for i in range(len(outs))]

    eta = eta_init
    v_norm = torch.norm(v)
    v_dir = v / v_norm if v_norm > 0 else v
    x_orig = parameters_to_vector(model.parameters()).detach().clone()

    while eta > 1e-12:
        # Restore original parameters
        vector_to_parameters(x_orig, model.parameters())

        # Apply candidate step (non-in-place update)
        params = []
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            new_p = p + eta * v_dir[offset:offset + numel].view_as(p)
            params.append(new_p)
            offset += numel

        # Update model parameters
        with torch.no_grad():
            for p, new_p in zip(model.parameters(), params):
                p.copy_(new_p)

        # Compute new loss (f_new) on the same fixed batch
        with torch.no_grad():
            outs = model(images)
            f_new = [criterion(outs[i], labels[:, i].long()).item() for i in range(len(outs))]

        # Armijo condition with preference
        lhs = np.array(f_new)
        rhs = np.array(f) + c1 * eta * np.array([g.dot(v_dir.cpu()) for g in grad])

        selected_lhs = []
        selected_rhs = []
        for i in range(len(pref)):
            if pref[i] < 0:
                selected_lhs.append(lhs[i])
                selected_rhs.append(rhs[i])

        if np.all(np.array(selected_lhs) <= np.array(selected_rhs)):
            # Restore original parameters before returning eta
            vector_to_parameters(x_orig, model.parameters())
            print(f"Backtracking found eta: {eta}")
            return eta  # Keep updated parameters

        eta *= beta  # Try smaller step

    # Restore original parameters if no suitable eta found
    print(f"Backtracking found eta: {eta}")
    vector_to_parameters(x_orig, model.parameters())
    return 0.0



   



def assign_grad(vector, model,g0,alpha,pref,criterion,trainloader, step_size = 0.01, c1=1e-4,normalize=True, prev_v= None,i =0,threshold_1 = 1e-4, threshold_2 = 1e-1,backtrack = False,  #threshold_2 = 1e-4 # step_size = 0.01
                   printed_corner= False,printed_extreme = False, printed_direction_impossible = False, printed_step = False):


    d = pref
    d_normalized = d / np.linalg.norm(d)
    d_y_proj = vector.cpu().numpy() #g0 @ vector     # or use: f(x0 + ε v_ud) - f(x0)
    d_y_proj = d_y_proj / np.linalg.norm(d_y_proj)

    alpha_normalized = alpha / np.linalg.norm(alpha)

    # (a) Dot product threshold
    condition_a = False
    if i > 0 :
        #print(d_y_proj, prev_v)
        dot_product = np.dot(d_y_proj,prev_v) #np.dot(d_y_proj,prev_v ) #np.dot(d_y_proj,d_normalized )
        condition_a = dot_product <= threshold_1

    max_alpha = np.linalg.norm(alpha_normalized, ord=np.inf)
    max_index = np.where(np.abs(alpha_normalized) == max_alpha)[0]#.item()
    max_index = max_index if isinstance(max_index, np.ndarray) else max_index # [0] 
    max_2_index = np.where(np.abs(alpha_normalized) == np.sort(alpha_normalized)[-2])[0]#.item()
    max_2_index = max_2_index if isinstance(max_2_index, np.ndarray) else max_2_index  #[0] 
    max_3_index = np.where(np.abs(alpha_normalized) == np.sort(alpha_normalized)[-3])[0]#.item()
    max_3_index = max_3_index if isinstance(max_3_index, np.ndarray) else max_3_index  #[0] 


    # (b) Sign oscillation
    condition_b = False
    if  i > 0 and prev_v is not None:
        #sign_change = np.sign(d_y_proj) == -np.sign(g0@prev_v)
        #condition_b = np.all(sign_change)  # all coordinates flipped sign

        sign_current = np.sign(d_y_proj)
        sign_prev = np.sign(prev_v)
        sign_flip = np.all((sign_current == -sign_prev) | (sign_prev == 0), axis=0)
        condition_b = np.all(sign_flip)


    # (c) Alpha ∞-norm threshold
    eps_2 = 1-threshold_2
    condition_c = max_alpha >= eps_2 

    while True:
        if condition_c and not printed_corner:
            print(f"⚠️ At corner,Direction {d} exceeds alpha ∞-norm: {max_alpha}; alpha: {alpha_normalized}")
            if max_alpha == 1 and not printed_extreme:
                print(f"At extreme point/minimum, take combination with {max_index + 1} to move away from corner")
                if check_combinations(pref, max_index) is None:
                    #True
                    return False, None
                
                
            elif 1-threshold_1 <= max_alpha < 1 and not printed_extreme:
                print(f"⚠️ Closer to an extreme point/minimum of objective {max_index + 1}, take combination with only objective {max_index + 1} to move closer or  to move away")
                if  (check_combinations(pref, max_2_index) is not None  and check_combinations(pref, max_3_index) is not None):
                    return False, None
                if (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_2_index) is not None) or (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_3_index) is not None) :
                    with torch.no_grad():
                        offset = 0
                        """if normalize:
                            """
                        if backtrack:
                            eta = backtracking_step(model, criterion, trainloader,vector, g0, eta_init=step_size, beta=0.5, c1=c1, pref=pref)
                        else: 
                            #vector = vector / vector.norm()
                            eta = step_size

                        vector = vector / vector.norm()
                        for p in model.parameters():
                            numel = p.numel()
                            p.add_(eta * vector[offset:offset + numel].view_as(p))
                            offset += numel
                    return True, d_y_proj
                #if (check_combinations(pref, max_index) is not None and check_combinations(pref, max_2_index) is not None):
                #    return False, None
                return True, d_y_proj
               

            elif 1-threshold_1 > max_alpha >= eps_2 and not printed_corner:
                #print(f"⚠️ At a corner of the pareto set, avoid a combination of objective {max_index + 1} and {max_3_index+1}, but take a combination with objective {max_index + 1} to strickly move away from the corner")
                print(f"⚠️ At a corner of the pareto set, avoid a combination of objective {max_index + 1} and {max_2_index+1}, but take a combination with objective {max_index + 1}  or {max_2_index+1} to strickly move away from the corner")
                if  (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_2_index) is not None):
                    return False, None
                if (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_3_index) is not None) or (check_combinations(pref, max_2_index) is not None  and check_combinations(pref, max_3_index) is not None) :
                    with torch.no_grad():
                        offset = 0
                        """if normalize:
                            vector = vector / vector.norm()"""
                        if backtrack:
                            eta = backtracking_step(model, criterion, trainloader,vector, g0, eta_init=step_size, beta=0.5, c1=c1, pref=pref)
                        else: 
                            #vector = vector / vector.norm()
                            eta = step_size

                        vector = vector / vector.norm()
                        for p in model.parameters():
                            numel = p.numel()
                            p.add_(eta * vector[offset:offset + numel].view_as(p))
                            offset += numel
                    return True, d_y_proj
                    
           
                return True, d_y_proj
            
            else: 
                print(f"❌ At extreme point/minimum direction impossible")
                return False, None
            
            if condition_a and not printed_direction_impossible:
                print(f"❌ Direction {d} impossible -- (dot too small): {dot_product}")
                return False, None
            
            else:
                if not printed_step:
                    print("Taking step")
                with torch.no_grad():
                    offset = 0
                    """if normalize:
                        vector = vector / vector.norm()"""
                    if backtrack:
                        eta = backtracking_step(model, criterion, trainloader,vector, g0, eta_init=step_size, beta=0.5, c1=c1, pref=pref)
                    else: 
                        #vector = vector / vector.norm()
                        eta = step_size
                    vector = vector / vector.norm()
                    for p in model.parameters():
                        numel = p.numel()
                        p.add_(eta * vector[offset:offset + numel].view_as(p))
                        offset += numel
                return True, d_y_proj
                    
            """elif condition_a and not printed_direction_impossible:
                print(f"❌ Direction {d} impossible -- (dot too small): {dot_product}")
                return False, None
            elif condition_b and not printed_corner:
                print(f"❌  Direction {d} oscillation -- sign flip: {d_y_proj}")
                return False, None"""

        else:
            with torch.no_grad():
                offset = 0
                """if normalize:
                    vector = vector / vector.norm()"""
                if backtrack:
                    eta = backtracking_step(model, criterion, trainloader,vector, g0, eta_init=step_size, beta=0.5, c1=c1, pref=pref)
                else: 
                    #vector = vector / vector.norm()
                    eta = step_size
                
                vector = vector / vector.norm()
                for p in model.parameters():
                    numel = p.numel()
                    p.add_(eta * vector[offset:offset + numel].view_as(p))
                    offset += numel
            return True, d_y_proj
 


def predictor_step(model,pref,criterion,count,linear_op_template,trainloader, jacobian_trainiter, prev_v = None, step_size = 0.01, n_obj=3,maxiter=100, momentum = 0.9 , printed_corner= False,
                   printed_extreme = False, printed_direction_impossible = False, printed_step = False, backtrack = False ):

    
    vs =[]
    
    # initalize momentum buffer
    jacobians_buffer_tensor = compute_grads(model, criterion,trainloader,  jacobian_trainiter)   
    jacobians_buffer = jacobians_buffer_tensor.clone().detach().cpu().numpy()

    #grads= jacobians_buffer#
    grads= row_normalize_G(jacobians_buffer)  # normalize rows

    selected_grads = []
    selected_prefs = []
    for i in range(len(pref)):
        if pref[i] < 0:
            selected_grads.append(grads[i])
            selected_prefs.append(pref[i])

    if len(selected_grads) >= 1:
        G = np.vstack(selected_grads).T   # (n_shared, m_sel)
        m = G.shape[1]
        n = G.shape[0]

        # Quadratic subproblems
        P = G.T @ G
        Theta = np.zeros((m, m))
        D = np.zeros((n, m))

        for i in range(m):
            q = G.T @ G[:, i] 
            #print("q",q)
            theta_i = solve_theta_nnls(P, q)#.cpu().numpy()   # >= 0
            #print("theta_i",theta_i)
            Theta[:, i] = theta_i
            D[:, i] = G[:, i] + G @ theta_i    #+

        d = D @ np.array(selected_prefs)
    else:
        # fallback: weighted sum of all grads
        d= grads.T @ pref   # (n,)

    
    alpha_buffer = compute_alpha(jacobians_buffer_tensor) # Entire 623 batches for UCI dataset with a match size of 256
    B = beta_basis(n_obj)  
    for b in  tqdm(B, desc='Span', leave=False):        
        model.train(True)

        # compute jacobians
        jacobians_tensor = compute_grads(model, criterion,trainloader, jacobian_trainiter, 1.0 / 4.0) # Use 1/4 of batches to compute jacobians ie 155 batches for UCI dataset with a match size of 256
        jacobians = jacobians_tensor.clone().detach().cpu().numpy()
        jacobians_buffer *= momentum
        jacobians_buffer += (1 - momentum) * jacobians
        jacobians = jacobians_buffer.copy()

        # compute alpha
        alpha = compute_alpha(jacobians_tensor)
        alpha_buffer *= momentum
        alpha_buffer += (1 - momentum) * alpha
        alpha = alpha_buffer.copy()

        # define rhs and x0
        rhs = np.squeeze(jacobians.T @ b)
        x0 = jacobians.mean(axis=0)
        
        # fill jacobians alpha rhs x0 to MINRES
        with linear_op_template.init(alpha) as linear_op:
            v, _ = minres(linear_op, rhs,x0=x0,  maxiter=maxiter) #
            
            vs.append(v)
            #
    V = np.stack(vs, axis=0)                          # shape (k, n)
    # 3) Orthonormalize rows via QR on V^T
    Q, R = np.linalg.qr(V.T, mode='reduced')          # Q: (n, k), R: (k, k)
    # Make R have positive diagonal (canonical QR)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign                                 # broadcast over columns
    v_orth = Q.T      

    proj_d = proj_general(v_orth, d)
    # optimize
    #optimizer.zero_grad()
    d = torch.as_tensor(proj_d, device=device)
    res, d_y_proj = assign_grad(d,model,jacobians_buffer,alpha_buffer, pref,criterion,trainloader,step_size= step_size, normalize=True, prev_v= prev_v, i = count,printed_corner= printed_corner,backtrack = backtrack,
                                printed_extreme = printed_extreme, printed_direction_impossible= printed_direction_impossible, printed_step= printed_step)
       
    #optimizer.step()
    return res, d_y_proj,alpha_buffer



def build_pref_vector(m, obj_indices, p):
    """Build a full preference vector of length m, where only the specified obj_indices have the given preference values p (negated)."""
    pref = np.zeros(m)
    for idx, val in zip(obj_indices, p):
        pref[idx] = -val
    return pref

def dm_desire():
    print("✅ Best corrector validation point is a new front)")
    while True:
        try:            
            lt = input(f"Do you want to continue with the new front? y/n: ").strip().lower() 
            if lt == "y" or lt == "yes":
                return True
            elif lt == "n" or lt == "no":
                return False
            else:                
                print("❌ Invalid input. Please enter y or n.")
        except ValueError:
            print("❌ Invalid input. Please enter y or n.")

def get_preference(max_obj=3):

    while True:
        try:
            objs =input("Select objectives (e.g. 1, 24, 135): ")
            obj_indices = [int(c) - 1 for c in objs]
            p = list(map(float, input(f"Enter {len(obj_indices)} preference values: ").split()))
            # Validate input length first
            if len(p) != len(obj_indices):
                raise ValueError(f"Expected {len(obj_indices)} values, got {len(p)}")
            pref = build_pref_vector(max_obj, obj_indices, p)
            return pref   # ✅ EXIT HERE

        except ValueError as e:
            print(f"❌ Invalid input: {e}")


def request_user_action(current_pref, current_max_runs):
    """
    User interaction handler.

    0 -> change preference
    1 -> rerun predictor (ask for max_runs)

    Returns:
        action: str ("change_pref" or "rerun_predictor")
        pref: np.ndarray
        max_runs: int or None
    """
    import numpy as np

    print("\nCurrent preference:", current_pref)
    print("Current max predictor runs:", current_max_runs)

    # ---- Select action ----
    while True:
        choice = input("Enter 0 to change preference, 1 to rerun predictor: ").strip()
        if choice in ("0", "1"):
            break
        print("❌ Invalid input. Please enter 0 or 1.")

    action = "change_pref" if choice == "0" else "rerun_predictor"

    # ---- Preference handling ----
    while True:
        keep = input("Keep current preference? (y/n): ").strip().lower()
        if keep in ("y", "n"):
            break
        print("❌ Invalid input. Please enter y or n.")

    if keep == "n":
        new_pref = input(
            "Enter new preference  (e.g. 0.2 0.5 0.3): "
        )
        pref = np.array([-float(x) for x in new_pref.split()])
    else:
        pref = current_pref

    # ---- Predictor rerun budget ----
    max_runs = None
    if action == "rerun_predictor":
        while True:
            try:
                max_runs = int(
                    input("Enter maximum number of predictor runs (positive integer): ")
                )
                if max_runs > 0:
                    break
                print("❌ Must be a positive integer.")
            except ValueError:
                print("❌ Invalid number.")

    return action, pref, max_runs




def get_file_directory():
    try:
        # Tries to get the directory of the current script (works in normal Python scripts)
        # We assume this file is a part of a larger project structure
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for environments like Jupyter/IPython where __file__ is undefined
        FILE_DIR = Path(os.getcwd()).resolve()
        
    # Set the path to save results: one directory level up, in a 'Results' folder
    # This structure mirrors the user's request
    file_path = FILE_DIR.parent / "Results"
    file_path.mkdir(parents=True, exist_ok=True)  # <-- Ensure folder exists
    return file_path

# Utilities ---------------------------------------------------------
def is_dominated(v, other_v,pref=None):
    """Return True if v is dominated by other_v (strict in at least one)."""
    if pref is not None:
        v_sel = [v[k] for k in range(len(pref)) if pref[k] < 0]
        other_v_sel = [other_v[k] for k in range(len(pref)) if pref[k] < 0]
        v = np.array(v_sel)
        other_v = np.array(other_v_sel)
        return np.all(other_v <= v) or np.any(other_v < v)

    v = np.array(v)
    other_v = np.array(other_v)
    return np.all(other_v <= v) and np.any(other_v < v)

def is_dom(v, other_v):
    """Return True if v is dominated by other_v (strict in at least one)."""
    v = np.array(v)
    other_v = np.array(other_v)
    return np.all(other_v <= v) and np.any(other_v < v)

def prune_archive(archive, typePred =False, pref = None):
    """
    Given archive = [(val_vec, ckpt_path), ...], return pruned nondominated archive.
    """
    pruned = []


    if typePred:
        # Keep only points that are nondominated w.r.t. preference
        for i, (v_i, p_i, d_i,acc_i) in enumerate(archive):
            dominated = False
            for j, (v_j, p_j, d_j,acc_j) in enumerate(archive):
                if i == j: 
                    continue
                # Check domination only on selected objectives
                v_i_sel = [v_i[k] for k in range(len(pref)) if pref[k] < 0]
                v_j_sel = [v_j[k] for k in range(len(pref)) if pref[k] < 0]
                if is_dom(v_i_sel, v_j_sel):
                    dominated = True
                    break
            if not dominated:
                pruned.append((v_i, p_i, d_i,acc_i))
        return pruned

    for i, (v_i, p_i, d_i,acc_i) in enumerate(archive):
        dominated = False
        for j, (v_j, p_j, d_j,acc_j) in enumerate(archive):
            if i == j: 
                continue
            if is_dom(v_i, v_j):
                dominated = True
                break
        if not dominated:
            pruned.append((v_i, p_i, d_i,acc_i))
    return pruned

def is_not_dominated(K, b):
    """
    Check if found losses b is not dominated by any row in matrix K (existing losses).
    Args:
        K: 2D numpy array of shape (n,3), where each column is a solution.
        b: 1D numpy array of shape (3,), representing the candidate solution.
    Returns:
        True if b is not dominated by any row in K, False otherwise.
    """
    for k in K:
        # Check if k dominates b
        if np.all(k <= b) and np.any(k < b):
            #print(f"Vector {b} is dominated by {k}")
            return False  # b is dominated by k
        
    return True  # b is not dominated by any row in K


def find_min_mean(pareto_array, pref):
    # Convert pref to a boolean mask where non-zero values are True
    mask = np.array(pref) < 0

    # Find the tuple with the minimum mean over the selected columns
    best_tuple = min(pareto_array, key=lambda x: np.mean(x[0][mask]))

    # Extract the best value and checkpoint
    best_val = best_tuple[0]
    best_ckpt = best_tuple[1]
    best_doc = best_tuple[2]

    return best_val, best_ckpt, best_doc

def save_or_copy(model, src_ckpt, dst_ckpt):
    src_ckpt = Path(src_ckpt)
    dst_ckpt = Path(dst_ckpt)

    # Ensure destination directory exists
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if src_ckpt.exists():
        # Avoid SameFileError
        if src_ckpt.resolve() != dst_ckpt.resolve():
            shutil.copy(src_ckpt, dst_ckpt)
    else:
        # Source checkpoint does not exist → save model directly
        torch.save(model.state_dict(), dst_ckpt)