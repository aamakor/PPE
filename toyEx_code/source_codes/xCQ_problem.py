# Run this part to get 3D version of the Pareto Set and front for 3 variables and 3 objectives.
import numpy as np
import matplotlib.pyplot as plt

from .method import * # reemove dot . to run separately
import cvxpy as cp
import tqdm
#np.set_printoptions(precision=7)
from scipy.spatial.distance import cdist



Q1 = np.array([[1, .7, 0], [.7, 1, 0], [0, 0, 1]])
Q2 = np.array([[1, 0, -.4], [0, 1, 0], [-.4, 0, 1]])
Q3 = np.array([[1, 0, 0], [0, 1, .7], [0, .7, 1]])

c1 = np.array([1, 0, 0])
c2 = np.array([0, 1.5, 0])
c3 = np.array([0, 0, 2])

#Uncomment to change the centers
"""c1 = np.array([1.0, 2.0, 0.0])
c2 = np.array([4.0, 2.0, 2.0])
c3 = np.array([1.0, 1.0, 3.0])"""



class x_c_problem(object):
    def __init__(self):
        self.n = 3
        self.m = 3

        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0

    def reset_count(self):
        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0


    def f(self, x):
        self.eval_f_cnt += 1
        return self.__f(x)

    def __f(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        f1 = .5*np.matmul(x - c1, np.matmul(Q1, x - c1))
        f2 = .5*np.matmul(x - c2, np.matmul(Q2, x - c2))
        f3 = .5*np.matmul(x - c3, np.matmul(Q3, x - c3))
        return ndarray([f1, f2, f3])

    def grad(self, x):
        self.eval_grad_cnt += 1
        return self.__grad(x)

    def __grad(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n
        
        grad_f1 = np.matmul(Q1, x - c1)#/np.linalg.norm(np.matmul(Q1, x - c1))
        grad_f2 = np.matmul(Q2, x - c2)#/np.linalg.norm(np.matmul(Q2, x - c2))
        grad_f3 = np.matmul(Q3, x - c3)#/np.linalg.norm(np.matmul(Q3, x - c3))
        
        return np.array([grad_f1, grad_f2, grad_f3])

    def hess(self, x):
        self.eval_hvp_cnt += 1
        return self.__hess(x)

    def __hess(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n

        # Hessian for f1 = (x[0] - 1)^4 + (x[1] - 1)^2 + (x[2] - 1)^2
        h1 = Q1

        # Hessian for f2 = (x[0] + 1)^2 + (x[1] + 1)^4 + (x[2] + 1)^2
        h2 = Q2
        # Hessian for f3 = (x[0] - 1)^2 + (x[1] + 1)^2 + (x[2] - 1)^4
        h3 = Q3
        #print("hessian shape", np.array([h1, h2, h3]))
        return np.array([h1, h2, h3])
    
    def hvp(self, x, alpha, v, reg = 1e-1):
        self.eval_hvp_cnt += 1
        h1, h2, h3 = self.hess(x)
        alpha = np.array(alpha).ravel()
        assert alpha.size == self.m
        v = np.array(v).ravel()
        assert v.size == self.n
        #H = np.array(alpha[0] * h1 + alpha[1] * h2 + alpha[2] * h3)
        Hv = np.array(alpha[0] * h1 @ v + alpha[1] * h2 @ v+ alpha[2] * h3 @ v) #H @ v
        return Hv + reg*v
    
    def hvp_q(self, x, alpha, isreg = False, reg = 1e-5):
        self.eval_hvp_cnt += 1
        h1, h2, h3 = self.hess(x)
        alpha = np.array(alpha).ravel()
        assert alpha.size == self.m

        H = np.array(alpha[0] * h1 + alpha[1] * h2 + alpha[2] * h3)
        if isreg == True:
            return H + reg * np.eye(H.shape[0])
        else:
            return H

    
    def weights_3d(self,n):
    
        Weights = np.zeros((3,int(n*(n+1)/2)))

        k = 0

        for i in range(n):
            for j in range(n-i):
                Weights[0, k] = i/(n-1)
                Weights[1, k] = j/(n-1)
                Weights[2, k] = 1 - i/(n-1) - j/(n-1)

                k = k+1
            
        return Weights

    def weighted_sum(self,d):

        Weights = self.weights_3d(d)
        d_ = int(d*(d+1)/2)
        X = np.zeros((3, d_))

        for j in range(d_):

            w1 = Weights[0, j]
            w2 = Weights[1, j]
            w3 = Weights[2, j]

            Q = w1*Q1 + w2*Q2 + w3*Q3
            c = w1*(Q1@c1) + w2*(Q2@c2) + w3*(Q3@c3)

            X[:, j] = np.linalg.solve(Q, c)

        return X.T

    
        
    def sample_pareto_set(self,n= 20):
        X = self.weighted_sum(n)
        #x = X[:, n]
        #********Uncomment to select the medoid(start point at the center) of the pareto set*********#
        """center = np.array([1/3, 1/3, 1/3])

        # Euclidean distances to the center
        distances = np.linalg.norm(X - center, axis=1)

        # Get indices of the closest points
        top_k = 1  # Number of points to select
        closest_idxs = np.argsort(distances)[:top_k]  # Negative for smallest

        # Extract selected alphas
        if top_k == 1:
            x = X[closest_idxs].ravel()
        else:
            x = X[closest_idxs]"""
        dist_matrix = cdist(X, X)
        total_dists = dist_matrix.sum(axis=1)
        medoid_index = np.argmin(total_dists)
        x = X[medoid_index]
        
        return x
    
 

    def plot_pareto_set(self, ax):
        x_pareto = self.weighted_sum(20).T
        # Plot the Pareto set in 3D decision space
        ax.scatter(x_pareto[0, :], x_pareto[1, :], x_pareto[2, :], c='red', alpha=0.6, label='Pareto Set')
        # Plot centers
        ax.scatter(c1[0], c1[1], c1[2], c='blue', s=50, label='C1')
        ax.scatter(c2[0], c2[1], c2[2], c='green', s=50, label='C2')
        ax.scatter(c3[0], c3[1], c3[2], c='black', s=50, label='C3')
        
        ax.set_xlabel(r"$𝓍_1$",fontsize = 20)
        ax.set_ylabel(r'$𝓍_2$',fontsize = 20)
        ax.set_zlabel(r'$𝓍_3$', fontsize = 20)
        ax.view_init(elev=30, azim=45)
        ax.legend()
        #ax.set_title(r'Decision Space $(\eta = 0.1)$', fontsize = 14)
        #plt.savefig("pareto_setxcq.png", dpi=300)
        ax.grid(True)


        
    def plot_pareto_front(self, ax, zorder = 1,label='Pareto front'):
        # Compute the Pareto front by evaluating f1, f2, and f3 on the Pareto set
        x = self.weighted_sum(20).T
  
        f1_vals = []
        f2_vals = []
        f3_vals = []
    
        for i in range(x.shape[1]):
            # Evaluate f1, f2, and f3
            f1_ = .5*np.matmul(x[:,i]- c1, np.matmul(Q1, x[:,i] - c1))
            f2_ = .5*np.matmul(x[:,i] - c2, np.matmul(Q2, x[:,i] - c2))
            f3_ = .5*np.matmul(x[:,i] - c3, np.matmul(Q3, x[:,i] - c3))
            f1_vals.append(f1_)
            f2_vals.append(f2_)
            f3_vals.append(f3_)
        f1_vals, f2_vals, f3_vals = np.array(f1_vals), np.array(f2_vals), np.array(f3_vals)
        # Plot the Pareto front in the objective space
        #ax.plot(f1_vals, f2_vals, f3_vals, 'k-.', label=label)
        ax.scatter(f1_vals, f2_vals, f3_vals, s = 20, label=label, zorder=zorder)
        ax.set_xlabel(r"$𝒇_1(𝓍)$", fontsize = 20)
        ax.set_ylabel(r"$𝒇_2(𝓍)$", fontsize = 20)
        ax.set_zlabel(r"$𝒇_3(𝓍)$",labelpad=10, fontsize = 20)
        #ax.text(1.5, 0.08, 0.9, r"$𝒇_3(𝓍)$", fontsize=20, rotation=180)
        ax.view_init(elev=20, azim=45)
        ax.legend()
        #ax.set_title(r'$Objective \ space \ (\eta = 0.1)$', fontsize = 14)
        #plt.savefig("pareto_frontxcq.png", dpi=300)
        ax.grid(True)

if __name__ == '__main__':
    # Check gradients.
    problem = x_c_problem()
    n, m = 3, 3
    #x0 = np.random.normal(size=n)
    x0 = 2*np.random.rand(n)-1
    for i in range(m):
        f = lambda x: problem.f(x)[i]
        g = lambda x: problem.grad(x)[i]
        check_grad(f, g, x0)
        h = lambda x : problem.hess(x)[i]
        check_hess(f, g, h, x0)

    # Check Pareto front.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    problem.plot_pareto_front(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    problem.plot_pareto_set(ax)

    plt.show()
    plt.close()
    
    
    
"""
# Run this part separately to get 2D version of the Pareto Set ie. 2 variables and 3 objectives.

import numpy as np
from common import *
import cvxpy as cp
import tqdm
#np.set_printoptions(precision=7)


Q1 = np.array([[1, .7], [.7, 1]])
Q2 = np.array([[1, 0], [0, 1]])
Q3 = np.array([[1, 0], [0, .7]])

c1 = np.array([1, 0])
c2 = np.array([0, 1.5])
c3 = np.array([0, 0])


class x_c_variant(object):
    def __init__(self):
        self.n = 2
        self.m = 3

        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0

    def reset_count(self):
        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0


    def f(self, x):
        self.eval_f_cnt += 1
        return self.__f(x)

    def __f(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        f1 = .5*np.matmul(x - c1, np.matmul(Q1, x - c1))
        f2 = .5*np.matmul(x - c2, np.matmul(Q2, x - c2))
        f3 = .5*np.matmul(x - c3, np.matmul(Q3, x - c3))
        return ndarray([f1, f2, f3])

    def grad(self, x):
        self.eval_grad_cnt += 1
        return self.__grad(x)

    def __grad(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n
        
        grad_f1 = np.matmul(Q1, x - c1)#/np.linalg.norm(np.matmul(Q1, x - c1))
        grad_f2 = np.matmul(Q2, x - c2)#/np.linalg.norm(np.matmul(Q2, x - c2))
        grad_f3 = np.matmul(Q3, x - c3)#/np.linalg.norm(np.matmul(Q3, x - c3))
        
        return np.array([grad_f1, grad_f2, grad_f3])

    def hess(self, x):
        self.eval_hvp_cnt += 1
        return self.__hess(x)

    def __hess(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n

        # Hessian for f1 = (x[0] - 1)^4 + (x[1] - 1)^2 + (x[2] - 1)^2
        h1 = Q1

        # Hessian for f2 = (x[0] + 1)^2 + (x[1] + 1)^4 + (x[2] + 1)^2
        h2 = Q2
        # Hessian for f3 = (x[0] - 1)^2 + (x[1] + 1)^2 + (x[2] - 1)^4
        h3 = Q3

        return np.array([h1, h2, h3])
    
    def hvp(self, x, alpha, v):
        self.eval_hvp_cnt += 1
        h1, h2, h3 = self.hess(x)
        alpha = np.array(alpha).ravel()
        assert alpha.size == self.m
        v = np.array(v).ravel()
        assert v.size == self.n
        return np.array(alpha[0] * h1 @ v + alpha[1] * h2 @ v+ alpha[2] * h3 @ v)
    

    

    
    def weights_3d(self,n):
    
        Weights = np.zeros((3,int(n*(n+1)/2)))

        k = 0

        for i in range(n):
            for j in range(n-i):
                Weights[0, k] = i/(n-1)
                Weights[1, k] = j/(n-1)
                Weights[2, k] = 1 - i/(n-1) - j/(n-1)

                k = k+1
            
        return Weights

    def weighted_sum(self,d):

        Weights = self.weights_3d(d)
        d_ = int(d*(d+1)/2)
        X = np.zeros((2, d_))

        for j in range(d_):

            w1 = Weights[0, j]
            w2 = Weights[1, j]
            w3 = Weights[2, j]

            Q = w1*Q1 + w2*Q2 + w3*Q3
            c = w1*(Q1@c1) + w2*(Q2@c2) + w3*(Q3@c3)

            X[:, j] = np.linalg.solve(Q, c)

        return X

    
        
    def sample_pareto_set(self,n):
        X = self.weighted_sum(20)
        x = X[:, n]
        return x

 
    def plot_pareto_set(self, ax):
        # Compute the Pareto set as the convex hull of the extreme points
        x_pareto = self.weighted_sum(20)
        # Plot the Pareto set in 3D decision space
        ax.scatter(x_pareto[0, :], x_pareto[1, :], c='red', alpha=0.6, label='Pareto Set')
        # Plot centers
        centers = np.vstack([c1, c2, c3])
        ax.scatter(centers[:, 0], centers[:, 1], c='blue', s=50, label='Centers')

        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        #ax.set_zlabel('$x_3$')
        ax.legend()
        ax.set_title('Pareto Set in Decision Space')
        ax.grid(True)


        
    def plot_pareto_front(self, ax, label='Pareto front'):
        # Compute the Pareto front by evaluating f1, f2, and f3 on the Pareto set
        x = self.weighted_sum(20)
  
        f1_vals = []
        f2_vals = []
        f3_vals = []
    
        for i in range(x.shape[1]):
            # Evaluate f1, f2, and f3
            f1_ = .5*np.matmul(x[:,i]- c1, np.matmul(Q1, x[:,i] - c1))
            f2_ = .5*np.matmul(x[:,i] - c2, np.matmul(Q2, x[:,i] - c2))
            f3_ = .5*np.matmul(x[:,i] - c3, np.matmul(Q3, x[:,i] - c3))
            f1_vals.append(f1_)
            f2_vals.append(f2_)
            f3_vals.append(f3_)
        f1_vals, f2_vals, f3_vals = np.array(f1_vals), np.array(f2_vals), np.array(f3_vals)
        # Plot the Pareto front in the objective space
        #ax.plot(f1_vals, f2_vals, f3_vals, 'k-.', label=label)
        ax.scatter(f1_vals, f2_vals, f3_vals, s = 20, label=label)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_zlabel('$f_3$')
        ax.legend()
        ax.set_title('Pareto Front in Objective Space')
        ax.grid(True)

if __name__ == '__main__':
    # Check gradients.
    problem = x_c_variant()
    n, m = 2, 3
    #x0 = np.random.normal(size=n)
    x0 = 2*np.random.rand(n)-1
    for i in range(m):
        f = lambda x: problem.f(x)[i]
        g = lambda x: problem.grad(x)[i]
        check_grad(f, g, x0)
        h = lambda x : problem.hess(x)[i]
        check_hess(f, g, h, x0)

    # Check Pareto front.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    problem.plot_pareto_front(ax)

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(1, 1, 1)
    problem.plot_pareto_set(ax)

    plt.show()
    plt.close()

"""