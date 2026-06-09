# Run this part to get 2D version of the Pareto Set and front for 2 variables and 2 objectives.
import numpy as np

from .method import * # Remove the dot if running outside a package
import cvxpy as cp
import tqdm

Q1 = np.array([[1, .7], [.7, 1]])
Q2 = np.array([[1, 0], [0, 1]])


c1 = np.array([4, 2,])
c2 = np.array([1, 2])



class x_c_problem(object):
    def __init__(self):
        self.n = 2
        self.m = 2

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
    
        return ndarray([f1, f2])

    def grad(self, x):
        self.eval_grad_cnt += 1
        return self.__grad(x)

    def __grad(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n
        
        grad_f1 = np.matmul(Q1, x - c1)
        grad_f2 = np.matmul(Q2, x - c2)

        
        return np.array([grad_f1, grad_f2])

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


        return np.array([h1, h2])
    
    def hvp(self, x, alpha, v, reg = 1e-5):
        self.eval_hvp_cnt += 1
        h1, h2 = self.hess(x)
        alpha = np.array(alpha).ravel()
        assert alpha.size == self.m
        v = np.array(v).ravel()
        assert v.size == self.n
        #return np.array(alpha[0] * h1 @ v + alpha[1] * h2 @ v)
        H = np.array(alpha[0] * h1 + alpha[1] * h2)
        Hv = H @ v
        return Hv + reg*v
    
 


    def weighted_sum(self,d):

        Weights = np.linspace(0,1,d)
        
        X = np.zeros((2, d))

        for j in range(d):

            w1 = Weights
            w2 = 1-w1

            Q = w1[j]*Q1 + w2[j]*Q2
            c = w1[j]*(Q1@c1) + w2[j]*(Q2@c2)

            X[:, j] = np.linalg.solve(Q, c)

        return X

    
        
    def sample_pareto_set(self,n=50):
        X = self.weighted_sum(n)
        n = np.random.randint(1,X.shape[1])
        x = X[:, n] #26
        return x


    def plot_pareto_set(self, ax):
        # Compute the Pareto set as the convex hull of the extreme points
        x_pareto = self.weighted_sum(50)
        # Plot the Pareto set in 2D decision space
        ax.plot(x_pareto[0, :], x_pareto[1, :],c='red', label='Pareto Set', linewidth=2)
        # Plot centers
        centers = np.vstack([c1, c2])
        ax.scatter(centers[:, 0], centers[:, 1], c='brown', s=50, label='Centers')
        ax.set_xlabel(r'$x_1$', fontsize = 20)
        ax.set_ylabel(r'$x_2$', fontsize = 20)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(1.5,3 )
        ax.set_xlim(0,5)
        ax.legend(fontsize = 14)
        #ax.set_title(r'$Decision \  space \ (\eta = 0.5)$', fontsize = 14)
        #ax.grid(True)


        
    def plot_pareto_front(self, ax, label='Pareto front'):
        # Compute the Pareto front by evaluating f1, f2, and f3 on the Pareto set
        x = self.weighted_sum(50)
  
        f1_vals = []
        f2_vals = []
    
    
        for i in range(x.shape[1]):
            # Evaluate f1, f2, and f3
            f1_ = .5*np.matmul(x[:,i]- c1, np.matmul(Q1, x[:,i] - c1))
            f2_ = .5*np.matmul(x[:,i] - c2, np.matmul(Q2, x[:,i] - c2))
            f1_vals.append(f1_)
            f2_vals.append(f2_)
        f1_vals, f2_vals = np.array(f1_vals), np.array(f2_vals)
        # Plot the Pareto front in the objective space
        #ax.plot(f1_vals, f2_vals, f3_vals, 'k-.', label=label)
        ax.scatter(f1_vals, f2_vals, s = 20, label=label)
        ax.set_xlabel(r'$f_1$', fontsize = 20)
        ax.set_ylabel(r'$f_2$', fontsize = 20)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize = 14)
        #ax.set_title(r'$Objective \ space \ (\eta = 0.5)$', fontsize = 14)
        #ax.grid(True)

"""if __name__ == '__main__':
    # Check gradients.
    problem = x_c_problem()
    n, m = 2, 2
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
    ax = fig.add_subplot(1, 1, 1)
    problem.plot_pareto_front(ax)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    problem.plot_pareto_set(ax)

    plt.show()
    plt.close()"""