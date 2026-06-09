
import numpy as np

from .method import *
import cvxpy as cp
import tqdm
import matplotlib.pyplot as plt


c1 = np.array([1.0,2.0,1.0])
c2 = np.array([4.0,2.0,1.0])
c3 = np.array([1.0,1.0,1.0])

#Uncomment to change the centers

"""
c1 = np.array([1, 0, 0])
c2 = np.array([0, 1.5, 0])
c3 = np.array([0, 0, 2])
"""

class x_c_problem(object):
    def __init__(self):
        self.m = 3
        self.n = len(c1)
        

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
        x = np.array(x).ravel()
        assert x.size == self.n
        f1 = 0.5 * (np.linalg.norm(x-c1, 2 ))**2
        f2 = 0.5 * (np.linalg.norm(x-c2, 2 ))**2
        f3 = 0.5 * (np.linalg.norm(x-c3, 2 ))**2
        return np.array([f1, f2, f3])

    def grad(self, x):
        self.eval_grad_cnt += 1
        return self.__grad(x)

    def __grad(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n
        
        # Gradient vector of f1
        grad_f1 = x - c1
        # Gradient vector of f2
        grad_f2 = x - c2
        # Gradient vector of f3
        grad_f3 = x - c3
        return np.array([grad_f1, grad_f2, grad_f3])

    def hess(self, x):
        self.eval_hvp_cnt += 1
        return self.__hess(x)

    def __hess(self, x):
        x = np.asarray(x).ravel()
        assert x.size == self.n

        # Hessian for f1 = (x[0] - 1)^4 + (x[1] - 1)^2 + (x[2] - 1)^2
        h1 = np.zeros((self.n, self.n))
        h1[0, 0] = 1
        h1[1, 1] = 1
        h1[2, 2] = 1

        # Hessian for f2 = (x[0] + 1)^2 + (x[1] + 1)^4 + (x[2] + 1)^2
        h2 = np.zeros((self.n, self.n))
        h2[0, 0] = 1
        h2[1, 1] = 1
        h2[2, 2] = 1

        # Hessian for f3 = (x[0] - 1)^2 + (x[1] + 1)^2 + (x[2] - 1)^4
        h3 = np.zeros((self.n, self.n))
        h3[0, 0] = 1
        h3[1, 1] = 1
        h3[2, 2] = 1

        return np.array([h1, h2, h3])
    
    def hvp(self, x, alpha, v, reg = 1e-5):
        self.eval_hvp_cnt += 1
        h1, h2, h3 = self.hess(x)
        alpha = np.array(alpha).ravel()
        assert alpha.size == self.m
        v = np.array(v).ravel()
        assert v.size == self.n
        #H = np.array(alpha[0] * h1 + alpha[1] * h2 + alpha[2] * h3)
        Hv = np.array(alpha[0] * h1 @ v + alpha[1] * h2 @ v+ alpha[2] * h3 @ v)#H @ v
        return Hv + reg*v#


    
    def lambdas_3d(self,n):
    
        Weights = np.zeros((3,int(n*(n+1)/2)))

        k = 0

        for i in range(n):
            for j in range(n-i):
                Weights[0, k] = i/(n-1)
                Weights[1, k] = j/(n-1)
                Weights[2, k] = 1 - i/(n-1) - j/(n-1)

                k = k+1
        #print(Weights.T)    
        return Weights.T

    
        
    def sample_pareto_set(self,num=50):

        lambdas = self.lambdas_3d(num)
        #print(lambdas)
        #nb = np.random.choice(1000)
        #selected = lambdas[nb]
        
        #print(lambdas)
        #print("lambda",selected) 
        # Compute distances from center (1/3, 1/3, 1/3)
        center = np.array([1/3, 1/3, 1/3])

        # Euclidean distances to the center
        distances = np.linalg.norm(lambdas - center, axis=1)

        # Get indices of the closest points
        top_k = 1  # Number of points to select
        closest_idxs = np.argsort(distances)[:top_k]  # Negative for smallest

        # Extract selected alphas
        if top_k == 1:
            selected = lambdas[closest_idxs].ravel()
        else:
            selected = lambdas[closest_idxs]

        #print("lambda",selected) 

        return np.array([selected[0]*c1 + selected[1]*c2 + selected[2]*c3]).ravel()


    def plot_pareto_set(self, ax):
        # Compute the Pareto set as the convex hull of the extreme points
        lambdas = self.lambdas_3d(15)
        #print(len(lambdas))

        #print(lambdas.shape)
        X = [l[0]*c1 + l[1]*c2 + l[2]*c3 for l in lambdas]
        
        x_pareto = np.array(X)

        #print(x_pareto)
        # Plot the Pareto set in 3D decision space
        ax.scatter(x_pareto[:, 0], x_pareto[:, 1], x_pareto[:, 2], c='red', alpha=0.5, label='Pareto Set',zorder=1)
        # Plot centers
        ax.scatter(c1[0], c1[1], c1[2], c='blue', s=50, label='C1',zorder=2)
        ax.scatter(c2[0], c2[1], c2[2], c='green', s=50, label='C2',zorder=2)
        ax.scatter(c3[0], c3[1], c3[2], c='black', s=50, label='C3',zorder=2)
        
        ax.set_xlabel(r"$𝓍_1$",fontsize = 20)
        ax.set_ylabel(r'$𝓍_2$',fontsize = 20)
        ax.set_zlabel(r'$𝓍_3$', fontsize = 20)
        ax.legend()
        ax.set_title('Pareto Set in Decision Space')
        plt.savefig("pareto_setxc.png", dpi=300)
        ax.grid(True)


        
    def plot_pareto_front(self, ax, label='Pareto front', zorder=1, type = None):
        # Compute the Pareto front by evaluating f1, f2, and f3 on the Pareto set
        lambdas = self.lambdas_3d(50) #10

        X = [l[0]*c1 + l[1]*c2 + l[2]*c3 for l in lambdas]
        x = np.array(X)
        f1_vals = []
        f2_vals = []
        f3_vals = []
    
        for i in range(x.shape[0]):
            # Evaluate f1, f2, and f3
            f1_ = 0.5 * (np.linalg.norm(x[i,:]-c1, 2 ))**2
            f2_ = 0.5 * (np.linalg.norm(x[i,:]-c2, 2))**2
            f3_ = 0.5 * (np.linalg.norm(x[i,:]-c3, 2 ))**2
            f1_vals.append(f1_)
            f2_vals.append(f2_)
            f3_vals.append(f3_)
        f1_vals, f2_vals, f3_vals = np.array(f1_vals), np.array(f2_vals), np.array(f3_vals)
        # Plot the Pareto front in the objective space
        #ax.plot(f1_vals, f2_vals, f3_vals, 'k-.', label=label)
    
        if type == 12:
            ax.plot(f1_vals, f2_vals, 'b--', label='Pareto front',zorder=zorder)
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_title('Pareto Front (Objective Space)')
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()

        elif type == 13:
            ax.plot(f1_vals, f3_vals, 'b--', label='Pareto front',zorder=zorder)
            ax.set_xlabel('f1')
            ax.set_ylabel('f3')
            ax.set_title('Pareto Front (Objective Space)')
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
        elif type == 23:
            ax.plot(f2_vals, f3_vals, 'b--', label='Pareto front',zorder=zorder)
            ax.set_xlabel('f2')
            ax.set_ylabel('f3')
            ax.set_title('Pareto Front (Objective Space)')
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()

        else:
            ax.scatter(f1_vals, f2_vals, f3_vals,alpha=0.2,marker='^',label=label,zorder=zorder,depthshade=False) # s = 20,
            ax.set_xlabel(r"$𝒇_1(𝓍)$", fontsize = 20)
            ax.set_ylabel(r"$𝒇_2(𝓍)$", fontsize = 20)
            ax.set_zlabel(r"$𝒇_3(𝓍)$",labelpad=10, fontsize = 20)
            #ax.text(1.5, 0.08, 0.9, r"$𝒇_3(𝓍)$", fontsize=20, rotation=180)
            #ax.view_init(elev=30, azim=60)
            ax.legend()
            ax.set_title('Pareto Front in Objective Space')
            plt.savefig("pareto_frontxc.png", dpi=300)
            ax.grid(True)

if __name__ == '__main__':

    problem = x_c_problem()
    #print(problem.lambdas_3d(2).shape)
    # Check gradients.
    problem.sample_pareto_set(num=50)
    
    m,n = 3, 3
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




