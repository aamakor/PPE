from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from .method import *
import cvxpy as cp
import tqdm


class MultiObjectiveProblem(ABC):
    def __init__(self, n_var, n_obj):
        self.n = n_var
        self.m = n_obj
        self.eval_cnt = 0
        self.grad_cnt = 0
        self.hvp_cnt = 0

    @abstractmethod
    def evaluate(self, x):
        pass

    # ---------- public API ----------
    def grad(self, x): #jacobian(self, x):
        return self._jacobian_fd(x)

    def hvp(self, x, alpha, v, reg=1e-6):
        return self._hvp_fd(x, alpha, v, reg)

    # ---------- finite-difference fallback ----------
    def _jacobian_fd(self, x, eps=1e-6):
        self.grad_cnt += 1

        x = np.asarray(x, dtype=float).ravel()
        J = np.zeros((self.m, self.n))

        eps = eps * (1.0 + np.linalg.norm(x))

        for i in range(self.n):
            xp = x.copy()
            xm = x.copy()

            xp[i] += eps
            xm[i] -= eps

            fp = self.evaluate(xp.reshape(1, -1))
            fm = self.evaluate(xm.reshape(1, -1))

            J[:, i] = (fp - fm) / (2 * eps)

        return J


    def _hvp_fd(self, x, alpha, v, reg = 0.0, eps=None):
        self.hvp_cnt += 1
    
        x = np.asarray(x).ravel()
        v = np.asarray(v).ravel()
        alpha = np.asarray(alpha).ravel()

        if eps is None:
            eps = 1e-6*(np.linalg.norm(x) + 1.0) # default epsilon  To set epsilon= 1e-6 * (||x||_2 + 1) #i.e. being scale-aware

        g1 = alpha @ self.grad(x + eps * v)
        g2 = alpha @ self.grad(x - eps * v)
        Hv = (g1 - g2) / (2 * eps) + reg * v

        # Optional Tikhonov regularization
        if reg != 0.0:
            Hv += reg * v   
        return Hv
    




class QuadraticMO(MultiObjectiveProblem):
    def __init__(self, Qs, cs):
        self.Qs = Qs
        self.cs = cs
        super().__init__(n_var=Qs[0].shape[0], n_obj=len(Qs))

    def evaluate(self, x):
        self.eval_cnt += 1
        return np.array([
            0.5 * (x - c) @ Q @ (x - c)
            for Q, c in zip(self.Qs, self.cs)
        ])

    def grad(self, x): #jacobian(self, x):
        self.grad_cnt += 1
        return np.array([
            Q @ (x - c)
            for Q, c in zip(self.Qs, self.cs)
        ])

    def hvp(self, x, alpha, v, reg=1e-6):
        self.hvp_cnt += 1
        H = sum(a * Q for a, Q in zip(alpha, self.Qs))
        return H @ v + reg * v



class DTLZProblem(MultiObjectiveProblem):
    def __init__(self, dtlz):
        super().__init__(dtlz.n_var, dtlz.n_obj)
        self.problem = dtlz

    def evaluate(self, x):
        self.eval_cnt += 1
        x = np.asarray(x).reshape(1, -1)
        return self.problem.evaluate(x)[0]



