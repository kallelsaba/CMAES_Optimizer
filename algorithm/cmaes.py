import numpy as np
from scipy.linalg import eigh

class CMAES:
    """CMA-ES: Covariance Matrix Adaptation Evolution Strategy"""
    
    def __init__(self, dim=30, bounds=[-100, 100], max_eval=30000, seed=None):
        self.dim = dim
        self.bounds = bounds
        self.max_eval = max_eval
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialisation
        self.mean = np.random.uniform(bounds[0], bounds[1], dim)
        # sigma initial = 1/3 de l'étendue des bounds (valeur standard CMA-ES)
        self.sigma = (bounds[1] - bounds[0]) / 3.0
        self.C = np.eye(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        
        # Paramètres adaptatifs
        self.pop_size = int(4 + 3 * np.log(dim))
        self.mu = self.pop_size // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        # <CHANGE> Garder weights comme 1D array, ne pas reshaper
        self.weights = weights / np.sum(weights)
        
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        self.evals = 0
        self.best_fitness = float('inf')
        self.best_solution = None
        self.history = []
    
    def ask(self):
        """Génère population"""
        y = np.random.randn(self.pop_size, self.dim)
        x = self.mean + self.sigma * (self.B @ (self.D[:, None] * y.T)).T
        x = np.clip(x, self.bounds[0], self.bounds[1])
        return x
    
    def tell(self, solutions, fitness):
        """Mise à jour stratégie"""
        idx = np.argsort(fitness)
        
        if fitness[idx[0]] < self.best_fitness:
            self.best_fitness = fitness[idx[0]]
            self.best_solution = solutions[idx[0]].copy()
        
        self.history.append(self.best_fitness)
        
        old_mean = self.mean.copy()
        # <CHANGE> Utiliser dot product correctement avec 1D weights
        self.mean = self.weights @ solutions[idx[:self.mu]]
        
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma
        
        # Calcul de hsig avec protection contre division par zéro
        # Utiliser max(1, evals) pour éviter le cas où evals = 0
        generation = max(1, self.evals) / self.pop_size
        hsig_denom = np.sqrt(1 - (1 - self.cs) ** (2 * generation) + 1e-10)
        hsig = (np.linalg.norm(self.ps) / hsig_denom / self.chiN < 1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma
        
        artmp = (1 / self.sigma) * (solutions[idx[:self.mu]] - old_mean)
        # <CHANGE> Corriger le calcul de covariance avec weights aplatis
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (self.pc[:, None] @ self.pc[None, :] + (not hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)
        
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, B = eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-14))
        self.B = B
        
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
    
    def optimize(self, f):
        """Optimisation complète"""
        while self.evals < self.max_eval:
            solutions = self.ask()
            fitness = np.array([f(x) for x in solutions])
            self.evals += len(solutions)
            self.tell(solutions, fitness)
        
        return self.best_solution, np.array(self.history)