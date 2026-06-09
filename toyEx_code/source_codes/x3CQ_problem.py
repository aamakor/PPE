
import numpy as np
import matplotlib.pyplot as plt
from .method import *
import cvxpy as cp
import tqdm

Q1 = np.array([[1, .7, 0], [.7, 1, 0], [0, 0, 1]])
Q2 = np.array([[1, 0, -.4], [0, 1, 0], [-.4, 0, 1]])
c1 = np.array([1.0,2.0,1.0])
c2 = np.array([4.0,2.0,1.0])


class x_c_problem(object):
    def __init__(self):
        self.n = len(c1)
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
        x = np.array(x)
        f1 = 0.5*np.matmul(x - c1, np.matmul(Q1, x - c1))
        f2 = 0.5*np.matmul(x - c2, np.matmul(Q2, x - c2))
        return np.array([f1, f2])

    def grad(self, x):
        self.eval_grad_cnt += 1
        x = np.array(x)
        g1 = np.matmul(Q1, x - c1)
        g2 = np.matmul(Q2, x - c2)
        return np.array([g1, g2])


    def hess(self, x):
      
        return np.array([Q1, Q2])

    def hvp(self, x, alpha, v, reg = 1e-5):
        self.eval_hvp_cnt += 1
        h1, h2 = self.hess(x)
        v = np.array(v).ravel()
        H = np.array(alpha[0] * h1 + alpha[1] * h2)
        Hv = H @ v
        return Hv + reg*v# np.array(alpha[0] * h1 @ v + alpha[1] * h2 @ v)

    def sample_pareto_set(self, num=1):
        lambdas = np.random.rand(num)
        if num == 1:
            return np.array([(1 - l) * c1 + l * c2 for l in lambdas]).ravel()
        else:
            return np.array([(1 - l) * c1 + l * c2 for l in lambdas])

    


    def plot_pareto_set(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        lambdas = np.linspace(0, 1, 100)
        X = [(1 - l) * c1 + l * c2 for l in lambdas]
        x_pareto = np.array(X)
        
        ax.scatter(x_pareto[:,0], x_pareto[:, 1], x_pareto[:,2], c='blue', alpha=0.6, label='Pareto Set')
        ax.scatter(c1[0], c1[1], c1[2], c='red', s=50, label='C1')
        ax.scatter(c2[0], c2[1], c2[2], c='green', s=50, label='C2')
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        ax.legend()
        ax.set_title('Pareto Set in Decision Space')
        ax.grid(True)


    def plot_pareto_front(self, ax=None, label='Pareto front'):
        if ax is None:
            fig, ax = plt.subplots()

        lambdas = np.linspace(0, 1, 200)
        X = np.array([(1 - l) * c1 + l * c2 for l in lambdas])
        f1 = [0.5 * (x - c1).T @ Q1 @ (x - c1) for x in X]
        f2 = [0.5 * (x - c2).T @ Q2 @ (x - c2) for x in X]
        ax.plot(f1, f2, 'm-', label='Pareto front')
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_title('Pareto Front (Objective Space)')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()


if __name__ == '__main__':
    # Check gradients.
    problem = x_c_problem()
    fig, ax = plt.subplots()
    problem.plot_pareto_front(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    problem.plot_pareto_set(ax)

    plt.show()
    plt.close()