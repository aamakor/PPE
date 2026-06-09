#Implementing the predictor-corrector method for 3-objective problems.

import numpy as np
import cvxpy as cp
import torch
from numpy.linalg import qr, solve
from qpsolvers import solve_qp
from scipy.sparse.linalg import cg,gmres, LinearOperator, minres
from source_codes.min_norm_solver import *
from source_codes.method import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
import os
import pickle
np.set_printoptions(precision=8)
#torch.set_printoptions(precision=4)
np.random.seed(24)


#*********** problem 1 ***********# Uncomment the desired variant
#from source_codes.xC_problem import x_c_problem

#*********** problem 2 ***********# Uncomment the desired variant
from source_codes.xCQ_problem import x_c_problem



######################################################################################################################################

maxiter = 1000       # Maximum allowable number of iterations in MINRES. You can vary it from 1 to 3 to see how results improve with this parameter.
s = 0.1 #0.01             # Step size.
gamma = 0.9         # The decay factor in line search.
num_exp = 10        # How many times you want to repeat the experiment with a different seed.
# Hyperparameters for line search.
ths = 1e-5          # For detecting convergence. Smaller value takes longer to converge.
eta_init =  1        # Initial step size in line search.
c1 = 0              # For strong Wolfe condition. Cannot be negative. Typically 1e-4. Larger values requires more iterations in line search.



problem = x_c_problem()
m, n = problem.m, problem.n

def compute_alpha(g,m=3):
    alpha_ = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(alpha_.T @ g))
    constraints = [alpha_ >= 0, cp.sum(alpha_) == 1]
    alpha_prob = cp.Problem(objective, constraints)
    optimal_loss = alpha_prob.solve()
    alpha_ = ndarray(alpha_.value).ravel()
    return alpha_


def line_search(x, f, grad, d, eta, c1):
    x = ndarray(x).ravel()
    assert x.size == n
    f = ndarray(f).ravel()
    assert f.size == m
    grad = ndarray(grad)
    assert grad.shape == (m, n)
    d = ndarray(d).ravel()
    assert d.size == n
    while True:
        x_new = x + eta * d
        f_new = problem.f(x_new)
        if np.all([fi_new <= fi + c1 * gradi.dot(d) * eta for fi, gradi, fi_new in zip(f, grad, f_new)]):
            return eta
        eta *= gamma
        print("reduce eta to ", eta)

def mgda_optimize(x):
    x = ndarray(x).ravel()
    assert x.size == n
    x_iter = np.copy(x)
    while True:
        g_iter = problem.grad(x_iter)
        f_iter = problem.f(x_iter)
        alpha_iter = compute_alpha(g_iter)
        # Negative sign here because d must be a *descent* direction.
        d = -ndarray(alpha_iter.T @ g_iter).ravel()
        # Make sure they are indeed descent.
        for gi in g_iter:
            assert gi.dot(d) <= 0 or np.isclose(gi.dot(d), 0)
        # Termination condition 1: gradient is too small.
        if np.linalg.norm(d) < ths:
            return alpha_iter, x_iter
        eta = line_search(x_iter, f_iter, g_iter, d, eta_init, c1)
        x_iter += eta * d
        # Termination condition 2: change is too little. Effectively, this means eta is too small.
        if eta * np.linalg.norm(d) < ths:
            print("change too small, increase step size")
            return alpha_iter, x_iter 
        

        
class Predictor(object):
    def __init__(self, x, grad):
        self.x = x
        self.grad = grad
    
    
    def minres_solver(self, x0, alpha,b,  reg = 0.0):
        n = len(b)
        x, _ = minres(LinearOperator((n, n), matvec=lambda v: problem.hvp(x0, alpha, v,reg = reg), rmatvec=lambda v: problem.hvp(x0, alpha, v,reg=reg)), b, maxiter=maxiter)
        #print(f"MINRES info: {_}")
        return np.array(x)
    
    def cg_solver(self, x0, alpha,b,reg = 0.0):
        n = len(b)
        x, _ = cg(LinearOperator((n, n), matvec=lambda v: problem.hvp(x0, alpha, v,reg = reg)), b, maxiter=maxiter)
        #print(f"CG info: {_}")
        return np.array(x)

    # Simple Jacobi preconditioner: M^{-1} ≈ diag(1 / diag(H))
    #diag_approx = np.array([H_op(np.eye(1, n, i).ravel())[i] for i in range(n)])
    #M_inv = LinearOperator((n, n), matvec=lambda y: y / diag_approx)

    def gmres_solver(self, x0, alpha,b,reg = 0.0):
        n = len(b)
        x, info = gmres(LinearOperator((n, n), matvec=lambda v: problem.hvp(x0, alpha, v,reg = reg)), b, maxiter=maxiter)#
        #print(f"GMRES info: {info}")
        return np.array(x)

    def jac_q(self, x, alpha, reg=1e-5, eps=1e-6):
        grad_fun = lambda x: np.dot(alpha, problem.grad(x))
        g0 = grad_fun(x)
        JF = np.zeros_like(x)
        for i in range(len(x)):
            x_eps = np.array(x, copy=True)
            x_eps[i] += eps
            g_eps = grad_fun(x_eps)
            JF[i] = (g_eps - g0)[i] / eps
        return JF + reg * np.eye(len(x))

    def qr_solver(self,alpha,b,reg=0.0):
        # 1. Perform QR decomposition
        # H = QR, where Q is orthogonal and R is upper triangular.
        H_op =self.jac_q(x0, alpha,reg = reg) 
        Q, R = qr(H_op)
        # Multiplying by Q.T gives Q.T(QR)x = Q.T b, which simplifies to Rx = Q.T b.
        x = solve(R, Q.T @ b) 
        return x




    def beta_basis(self,m):
        """Construct n x k matrix with row-sum 0 and full row rank (works if n <= k-1)."""
        k = m-1
        if k > m:
            raise ValueError("Impossible: need n <= k-1 for full row rank with row-sum 0.")
        B = np.zeros((k, m), dtype=float)
        for i in range(k):
            B[i, i] = 1.0
            B[i, i+1] = -1.0
        return B

    def generate_span_vi(self, x0, alpha, reg, n_obj=3, solvers = "minres"):
        """
        Generates two orthogonal vectors v1, v2 in the span of beta1, beta2.
        Returns: (v1, v2) as a tuple of numpy arrays.
        """
        # Step 1: Generate two random vectors in R^3
        vi = []

        beta = self.beta_basis(n_obj)
        for b in beta:
            rhs = problem.grad( x0).T @ b  # shape: (n,)
            if solvers == "minres":
                v = self.minres_solver( x0, alpha, rhs, reg= reg)
            elif solvers == "gmres":
                v = self.gmres_solver(x0, alpha,rhs, reg= reg)
            elif solvers == "cg":
                v = self.cg_solver(x0, alpha,rhs, reg= reg)   
            elif solvers == "qr":
                v = self.qr_solver(alpha,rhs, reg= reg)

            v = v #if np.linalg.norm(v) == 0 else v/np.linalg.norm(v) 
            vi.append(v)

        # Compute reduced QR of v.T (shape 3 x 2 -> Q: 3x2, R: 2x2)
        vi = np.array(vi)  # shape k x n
        Q, R = np.linalg.qr(vi.T, mode='reduced')

        # Make R have positive diagonal (canonical QR)
        diag_sign = np.sign(np.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        Q = Q * diag_sign  # broadcast to columns

        # Orthonormal rows are the transpose of Q
        v_orth = Q.T  # shape k x n
        return v_orth
    
    def proj_general(self, V, d):
        V = np.asarray(V, float).T          # (n,k)
        d = np.asarray(d, float).ravel()  # (n,)
        M = V.T @ V              # (k,k)
        #print(V.shape, d.shape, M.shape)     
        if np.allclose(M, np.eye(M.shape[0])):
            return V @ (V.T @ d)  # (n,)  # V.T @ d    (k,)
        return V @ np.linalg.solve(M, V.T @ d)

    def compute_descent_direction(self, x0, alpha, grads, pref, solvers = "minres", reg =0):
        """
        V: (k, n) array of v_j vectors
        grad_f: (m, n) array of gradients ∇f_i(x)
        P: (m,) array of weights P_i
        Returns: optimal descent direction d ∈ ℝ^n
        """
        selected_grads = []
        selected_prefs = []
        for i in range(len(pref)):
            if pref[i] < 0:
                selected_grads.append(grads[i])
                selected_prefs.append(pref[i])
                
        if len(selected_grads) >= 1:
            G = np.vstack(selected_grads).T  # (n, k) matrix of selected gradients
            #assert G.shape[1] == len(selected_grads), "Number of selected gradients must match the number of columns in G."     

            m = G.shape[1]  # Number of selected gradients
            n = G.shape[0]  # Number of features
            
            P = np.matmul(G.T, G)
            P = 0.5 * (P + P.transpose())
            Theta = np.zeros((m, m))         # columns will be θ^i
            D = np.zeros((n, m))             # columns will be d_i

            for i in range(m):
                q = P[:, i]  #G.T @ G[:, i] #not P[i, :]
                theta_i = solve_qp(P, q, G = None, h = None, A =  None, b =  None, lb = np.zeros((m)), ub = None, solver="cvxopt")
                Theta[:, i] = theta_i
                D[:, i] = G[:, i] +  G @ theta_i

        
            d =  D @ np.array(selected_prefs)
            vi = self.generate_span_vi(x0, alpha, reg,grads.shape[0],solvers)  # Generate orthonormal basis in the span of beta vectors
            proj_d = self.proj_general(vi, d)
            #print("d", d)
            #print("proj_d", proj_d)
            return proj_d
        
        else:
            d = grads.T @ pref  # If no gradients are selected, use the full gradient
            # Solve for v
            vi = self.generate_span_vi(x0, alpha, reg,grads.shape[0],solvers)  # Generate orthonormal basis in the span of beta vectors
            proj_d = self.proj_general(vi, d)
            return proj_d 




def backtracking_line_search(problem, x, v, f, grad, eta_init=1.0, beta=0.5, c1=1e-4, pref=None):
    """
    Backtracking line search based on Armijo condition.
    Finds maximum eta in {1, 1/2, 1/4, ...} satisfying:
        F(x + eta*d) <= F(x) + c1 * eta * JF(x) * d
    """
    
    eta = eta_init
    while eta > 1e-12:
        selected_lhs = []
        selected_rhs = []
        x_new = x + eta * v  if np.linalg.norm(v) == 0 else x + eta *v / np.linalg.norm(v)
        f_new = problem.f(x_new)
        # Armijo-like condition (vectorized check)
        lhs = np.array(f_new)
        rhs = np.array(f) + c1 * eta * np.array([g.dot(v) for g in grad])
        
        for i in range(len(pref)):
            if pref[i] < 0:
                #print("selected objective ", pref[i], " lhs ", lhs[i], " rhs ", rhs[i])
                selected_lhs.append(lhs[i])
                selected_rhs.append(rhs[i])
        selected_lhs = np.array(selected_lhs)
        selected_rhs = np.array(selected_rhs)
        if np.all(selected_lhs <= selected_rhs):
            return eta
        eta *= beta  # halve step length (t = 1/2^j)
    return 0.0  # fallback if no eta satisfies condition


def check_combinations(arr,max_index, checkTwo = False, index2 = None):
    if arr[max_index] < 0:
        print(f"✅ Valid preference {arr}")
        return arr
    else:
        print(f"❌ change preference must include objective {max_index+1}")
    
    if checkTwo == True:
        if len(index2) > 1 :
            for i in index2:
                if arr[max_index] < 0 and arr[i] == 0:
                    return arr
                else:
                    print(f"❌ change preference must exclude objective {index2+1}")
    return None


def assign_grad(x0,vector, g0,alpha,pref, step_size = 0.01, normalize=True, prev_v= None,i =0,threshold_1 = 1e-5, threshold_2 = 1e-1,  #threshold_1 = 1e-4 # step_size = 0.01
                   printed_corner= False,printed_extreme = False, printed_direction_impossible = False, printed_step = False):


    d = pref
    d_normalized = d / np.linalg.norm(d)
    d_y_proj = vector #g0 @ vector     # or use: f(x0 + ε v_ud) - f(x0)
    d_y_proj = d_y_proj  if np.linalg.norm(d_y_proj) == 0 else  d_y_proj / np.linalg.norm(d_y_proj)  

    alpha_normalized = alpha # / np.linalg.norm(alpha)

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


    #sign_change = np.sign(d_y_proj) == -np.sign(d_normalized)
    #condition_b = np.all(sign_change)
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
                    return None, False, None
                
            elif 1-threshold_1 <= max_alpha < 1 and not printed_extreme:
                print(f"⚠️ Closer to an extreme point/minimum of objective {max_index + 1}, take combination with only objective {max_index + 1} to move closer or  to move away")
                if  (check_combinations(pref, max_2_index) is not None  and check_combinations(pref, max_3_index) is not None):
                    return None, False, None
                if (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_2_index) is not None) or (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_3_index) is not None) :
                    #pred_x = x0 + step_size * vector if np.linalg.norm(vector) == 0 else x0 + step_size *vector / np.linalg.norm(vector) 
                    f = problem.f(x0)
                    eta = backtracking_line_search(problem, x0, vector, f, g0, eta_init=step_size, beta=0.5, c1=1e-4, pref=pref)  
                    pred_x = x0 + eta * vector if np.linalg.norm(vector) == 0 else x0 + eta *vector / np.linalg.norm(vector) 
                    return pred_x,True, d_y_proj
        

            elif 1-threshold_1 > max_alpha >= eps_2 and not printed_corner:
                #print(f"⚠️ At a corner of the pareto set, avoid a combination of objective {max_index + 1} and {max_3_index+1}, but take a combination with objective {max_index + 1} to strickly move away from the corner")
                print(f"⚠️ At a corner of the pareto set, avoid a combination of objective {max_index + 1} and {max_2_index+1}, but take a combination with objective {max_index + 1}  or {max_2_index+1} to strickly move away from the corner")
                if  (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_2_index) is not None):
                    return None,False, None
                if (check_combinations(pref, max_index) is not None  and check_combinations(pref, max_3_index) is not None) or (check_combinations(pref, max_2_index) is not None  and check_combinations(pref, max_3_index) is not None) :
                    f = problem.f(x0)
                    eta = backtracking_line_search(problem, x0, vector, f, g0, eta_init=step_size, beta=0.5, c1=1e-4, pref=pref)  
                    pred_x = x0 + eta * vector if np.linalg.norm(vector) == 0 else x0 + eta *vector / np.linalg.norm(vector) 
                    return pred_x,True, d_y_proj
                
                #pred_x = x0 + step_size * vector if np.linalg.norm(vector) == 0 else x0 + step_size *vector / np.linalg.norm(vector) 
                f = problem.f(x0)
                eta = backtracking_line_search(problem, x0, vector, f, g0, eta_init=step_size, beta=0.5, c1=1e-4, pref=pref)  
                pred_x = x0 + eta * vector if np.linalg.norm(vector) == 0 else x0 + eta *vector / np.linalg.norm(vector) 
                return pred_x,True, d_y_proj
            
            else: 
                print(f"❌ At extreme point/minimum direction impossible")
                return None, False, None
            
            if condition_a and not printed_direction_impossible:
                print(f"❌ Direction {d} impossible -- (dot too small): {dot_product}")
                return None, False, None
            
            else:
                if not printed_step:
                    print("Taking step")
                #pred_x = x0 + step_size * vector if np.linalg.norm(vector) == 0 else x0 + step_size *vector / np.linalg.norm(vector)
                f = problem.f(x0)
                eta = backtracking_line_search(problem, x0, vector, f, g0, eta_init=step_size, beta=0.5, c1=1e-4, pref=pref)  
                pred_x = x0 + eta * vector if np.linalg.norm(vector) == 0 else x0 + eta *vector / np.linalg.norm(vector) 
                return pred_x, True, d_y_proj
                    
        #elif condition_a and not printed_direction_impossible:
        #    print(f"❌ Direction {d} impossible -- (dot too small): {dot_product}")
        #    return None, False, None
        elif condition_b and not printed_corner:
            print(f"❌  Direction {d} oscillation -- sign flip: {d_y_proj}")
            return None, False, None

        else:
            # Take a line search step
            f = problem.f(x0)
            eta = backtracking_line_search(problem, x0, vector, f, g0, eta_init=step_size, beta=0.9, c1=1e-4, pref=pref)  
            #print("ETA",eta)
            pred_x = x0 + eta * vector if np.linalg.norm(vector) == 0 else x0 + eta *vector / np.linalg.norm(vector) 
            # Termination condition 2: change is too little. Effectively, this means eta is too small.
            #if eta * np.linalg.norm(d) < ths:
            #    print("change too small, increase step size")
            #    return x_iter 
            return pred_x , True, d_y_proj
        




def predictor_step(x0, pref,count, prev_v= None,threshold_1 = 1e-4, threshold_2 = 1e-1, reg = 1e-1):
    g0 = problem.grad(x0)
    #print("g0: ",g0)
    # Predictor step
    predictor = Predictor(x0,g0)


    alpha_= compute_alpha(g0)
    alpha, _ = find_min_norm_element(torch.tensor(g0))
    
    #print("alphas: ", alpha)
    #print("alphasf: ", alpha_)
    norms = np.linalg.norm(g0, axis=1)
    g_normalized = g0 if norms.all() == np.zeros(len(norms)).all() else g0 / norms[:, np.newaxis]


    v = predictor.compute_descent_direction(x0, alpha, g_normalized , pref,solvers= "minres", reg=reg) #

    pred_x, success, d_y_proj = assign_grad(x0,v, g0,alpha,pref, step_size = s, prev_v= prev_v, i = count,threshold_1 = threshold_1, threshold_2 = threshold_2)

    return pred_x, success, d_y_proj



def corrector_step(x):
    alpha_c, x_new = mgda_optimize(x)
    return alpha_c, x_new 
    

def get_preference():

    while True:
        try:
            objs = int(input("Select objective to minimize 1, 2, 3, 12, 13, 23, 123 for objective predictor step: \n") )
        
            if objs ==1:
                p1 = float(input(f"enter preference value for obj{objs}: ") )
                pref = np.array([-p1,0,0])
                break
            
            elif objs ==2:
                p1 = float(input(f"enter preference value for obj {objs}: ") )
                pref = np.array([0,-p1,0])
                break

            elif objs ==3:
                p1 = float(input(f"enter preference value for obj {objs}: ") )
                pref = np.array([0,0,-p1])
                break  

            elif objs == 12 or objs == 13 or objs == 23 or objs == 123:
                try:
                    p = list(map(float, input(f"enter two or three preference values (obj{objs}): ").split()))

                    # Validate input length first
                    expected_count = len(str(objs))  # "12" → 2, "123" → 3, etc.
                    if len(p) != expected_count:
                        raise ValueError(f"Expected {expected_count} values, got {len(p)}")
                    
                    if objs == 12:
                        pref = np.array([-p[0], -p[1], 0])
                        break  
                    elif objs == 13:
                        pref = np.array([-p[0], 0, -p[1]])
                        break  
                    elif objs == 23:
                        pref = np.array([0 , -p[0], -p[1]])
                        break  
                    elif objs ==123:
                        pref = np.array([-p[0], -p[1], -p[2]])
                        break  
                    else:
                        raise ValueError("Input valid type and preference choice.")
                except ValueError as e:
                    print(f"❌ Invalid input. {e}  ")
            else:
                print("❌ input a valid objective choice and preference") 

        except ValueError:
            print("❌ Invalid input. Please enter valid preference again ")

    return pref

####### MULTI-TASK CM Training #######
if __name__ == "__main__":


    # Initialize lists to store data
    prefs = []
    fi_preds = []
    fi_corrs = []
    pred_points = []
    corr_points = []
    alphas = []
    max_iterations = 20 #50 #20

    colors = plt.cm.viridis(np.linspace(0, 1, max_iterations))  # Color map for different preferences

    fig_pf = plt.figure(figsize=(10, 10))
    ax_pf = fig_pf.add_subplot(111, projection='3d')
    # Set up interactive mode
    plt.ion()

    problem.plot_pareto_front(ax_pf, label='Pareto front',zorder=1)
    # Generate the initial Pareto optimal point.
    x_start = problem.sample_pareto_set(20) # np.array([0.24537, 0.89436, 0.15772])# problem.sample_pareto_set() #np.array([2.8883,  2.88756, 2.71097])# np.array([4,  1, 1])# np.array([0.24537, 0.89436, 0.15772]) #
    alpha0, x0 = mgda_optimize(x_start) # Ensure it is an optimal point
    f0 = problem.f(x0)

    alphas.append(alpha0) 
    print("initial f0 ", f0) 
    print("initial x0 ",x0)

    #print("gradfn", grad_fd(x0))
    #print("grad_analytical", problem.grad(x0))

    
    #problem.plot_pareto_front(ax_pf, label='Pareto front')
    ax_pf.scatter(f0[0], f0[1],f0[2], c='black', label='Pareto optimal $f(x^*)$', s=100,zorder=2)
    ax_pf.set_xlabel('f1')
    ax_pf.set_ylabel('f2')
    ax_pf.set_zlabel('f3')

    # Pause to allow plot to update
    plt.pause(0.1)

    prev_v = None
    count = 0

    while count < max_iterations: #while True:
        print(f"\n=== Predictor-Corrector iteration {count} ===")

        pref = get_preference()

        if count == 0:
            prev_v= None
            xt = x0
        else:
            prev_v= prev_success if prev_success is not None else prev
            xt = corr_x

        if count > 0:
            print("new_f0 ", fi_corr) 
        
        pred_x, success, prev = predictor_step(xt, pref, count, prev_v, reg=1e-1, threshold_1=1e-6, threshold_2=1e-1) # 1e-2

        if not success:
            print(f"🔁 Restarting predictor loop with new pref")
            continue

        # If success is True, continue 
        prev_success = prev  # Update prev_success only if success is True
        fi_pred = problem.f(pred_x) 
        print("fp ", fi_pred) 
        if count == 0:
            ax_pf.plot([f0[0], fi_pred[0]], [f0[1], fi_pred[1]], [f0[2],fi_pred[2]],"-X", c="blue",label = "predictor",linewidth=2,zorder=2) 

        else:
            ax_pf.plot([fi_pred[0], fi_corr[0]], [fi_pred[1], fi_corr[1]], [fi_pred[2],fi_corr[2]],"-X", c="blue",linewidth=2,zorder=2) # ,label = "predictor",

        alpha_c, corr_x = corrector_step(pred_x)
        fi_corr = problem.f(corr_x)
        #xt = corr_x
        #print("x0 :", xt) 
        print("fc ", fi_corr) 
        #print("mgda_optimize fc ",problem.f(mgda_optimize(pred_x)))
        if count == 0:
            ax_pf.plot([fi_corr[0], fi_pred[0]], [fi_corr[1], fi_pred[1]], [fi_corr[2], fi_pred[2]],"-o",  color='red', label = "corrector",linewidth=2,zorder=2)
        else:
            ax_pf.plot([fi_corr[0], fi_pred[0]], [fi_corr[1], fi_pred[1]], [fi_corr[2], fi_pred[2]],"-o", color='red',linewidth=2,zorder=2) #, label = "corrector",
        
        ax_pf.set_xlabel('f1')
        ax_pf.set_ylabel('f2')
        ax_pf.set_zlabel('f3')

        # Pause to allow plot to update
        plt.pause(0.1)

        # Store data
        prefs.append(pref)
        alphas.append(alpha_c) 
        fi_preds.append(fi_pred)
        fi_corrs.append(fi_corr)
        pred_points.append(pred_x)
        corr_points.append(corr_x)
        
        count += 1
       
        ax_pf.legend()
    plt.show()

    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    #file_path = FILE_DIR.parent / "toyEx_code/toy_results/ResultsMathsxc"
    file_path = FILE_DIR.parent / "toyEx_code/toy_results/ResultsMathsxqc"
    file_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists

    with open(file_path / f'first_result_{count}.pkl', 'wb') as f:
        pickle.dump((alphas,f0,fi_preds,fi_corrs,prefs), f)

    


   