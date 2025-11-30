import numpy as np

# Utilisation du benchmark CEC2017 officiel
# Source: https://github.com/tilleyd/cec2017-py
try:
    from cec2017.functions import all_functions
    CEC2017_OFFICIAL = True
except ImportError:
    CEC2017_OFFICIAL = False
    # Note: Pour installer le benchmark officiel:
    # pip install git+https://github.com/tilleyd/cec2017-py.git


class CEC2017:
    """CEC2017 Benchmark Functions
    
    Utilise le benchmark officiel CEC2017 si disponible.
    Source: https://github.com/tilleyd/cec2017-py
    
    Les 30 fonctions sont divisées en 4 catégories:
    - F1-F3: Unimodales
    - F4-F10: Multimodales  
    - F11-F20: Hybrides
    - F21-F30: Composées
    
    Dimension: 30 (par défaut)
    Bounds: [-100, 100]
    """
    
    def __init__(self, fid, dim=30):
        if fid < 1 or fid > 30:
            raise ValueError("fid doit être entre 1 et 30")
        
        # Note: F2 a été retirée du benchmark officiel CEC2017
        # Le package cec2017 gère cela en interne
        self.fid = fid
        self.dim = dim
        self.bounds = [-100, 100]
        
        if CEC2017_OFFICIAL:
            # Utiliser le benchmark officiel
            # all_functions retourne une liste de fonctions (indices 0-29 pour F1-F30)
            self._official_func = all_functions[fid - 1]
        else:
            # Fallback sur implémentation simplifiée
            np.random.seed(fid * 123)
            self.offset = np.random.uniform(-80, 80, self.dim)
            self._init_rotation()
    
    def _init_rotation(self):
        """Initialise une matrice de rotation orthogonale (fallback)"""
        np.random.seed(self.fid * 456)
        A = np.random.randn(self.dim, self.dim)
        Q, R = np.linalg.qr(A)
        self.rotation = Q
    
    def __call__(self, x):
        """Évalue la fonction CEC2017"""
        x = np.asarray(x, dtype=np.float64)
        
        if x.shape[0] != self.dim:
            raise ValueError(f"x doit être de dimension {self.dim}, reçu {x.shape[0]}")
        
        if CEC2017_OFFICIAL:
            # Le benchmark officiel attend un array 2D (1 x dimension)
            x_2d = x.reshape(1, -1)
            result = self._official_func(x_2d)
            # Retourne un scalaire
            return float(result[0]) if hasattr(result, '__len__') else float(result)
        else:
            # Fallback sur implémentation simplifiée
            return self._fallback_eval(x)
    
    def _fallback_eval(self, x):
        """Évaluation fallback si le package officiel n'est pas disponible"""
        x_shifted = x - self.offset
        
        if self.fid <= 3:
            return self._unimodal(x_shifted)
        elif self.fid <= 10:
            return self._multimodal(x_shifted)
        elif self.fid <= 20:
            return self._hybrid(x_shifted)
        else:
            return self._composition(x_shifted)
    
    def _unimodal(self, x):
        """Fonctions unimodales F1-F3"""
        if self.fid == 1:
            return np.sum(x ** 2) + 100
        elif self.fid == 2:
            n = len(x)
            weights = 10 ** (6 * np.arange(n) / (n - 1))
            return np.sum(weights * x ** 2) + 200
        else:
            return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2) + 300
    
    def _multimodal(self, x):
        """Fonctions multimodales F4-F10"""
        n = len(x)
        if self.fid == 4:
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2) + 400
        elif self.fid == 5:
            return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10) + 500
        elif self.fid == 9:
            return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e + 900
        else:
            return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, n+1)))) + 1 + 100*self.fid
    
    def _hybrid(self, x):
        """Fonctions hybrides F11-F20"""
        n = len(x)
        n1, n2 = n // 3, 2 * n // 3
        f1 = np.sum(x[:n1] ** 2)
        f2 = np.sum(x[n1:n2]**2 - 10*np.cos(2*np.pi*x[n1:n2]) + 10)
        f3 = np.sum(100*(x[n2+1:] - x[n2:-1]**2)**2 + (x[n2:-1]-1)**2) if n2 < n-1 else 0
        return f1 + f2 + f3 + 1000 + 100 * self.fid
    
    def _composition(self, x):
        """Fonctions composées F21-F30"""
        n = len(x)
        f1 = np.sum(x ** 2)
        f2 = np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)
        f3 = -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e
        return (f1 + f2 + f3) / 3 + 2000 + 100 * self.fid


def get_function_info(fid):
    """Retourne les informations sur une fonction CEC2017"""
    categories = {
        (1, 3): ("Unimodale", "Fonctions avec un seul optimum global"),
        (4, 10): ("Multimodale", "Fonctions avec plusieurs optima locaux"),
        (11, 20): ("Hybride", "Combinaison de plusieurs fonctions de base"),
        (21, 30): ("Composée", "Composition complexe de fonctions")
    }
    
    for (start, end), (cat, desc) in categories.items():
        if start <= fid <= end:
            return {
                "fid": fid,
                "category": cat,
                "description": desc,
                "bounds": [-100, 100],
                "dimension": 30,
                "official": CEC2017_OFFICIAL
            }
    
    return None