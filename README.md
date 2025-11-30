# CMA-ES Interactive Optimizer

Dashboard interactif Streamlit pour explorer l'optimisation continue avec **CMA-ES** et le benchmark **CEC2017**.

## 📚 Projet Académique

**Matière** : Méthodes Heuristiques et Métaheuristiques  

**Équipe** :
- Eya Zouch  
- Oumayma Khlif  
- Saba Kallel  

---

## 🚀 Installation avec Anaconda

> ⚠️ **Important** : Utilisez **Anaconda** (ou Miniconda) pour éviter les erreurs de compatibilité. L'utilisation de `venv` peut causer des problèmes avec certaines dépendances scientifiques.

### 1. Prérequis

- [Anaconda](https://www.anaconda.com/download) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### 2. Cloner le projet

```bash
git clone https://github.com/eyazouch/cmaes-optimizer.git
cd cmaes-optimizer
```

### 3. Créer l'environnement Conda

```bash
conda create -n cmaes python=3.10 -y
conda activate cmaes
```

### 4. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Installer le benchmark CEC2017 officiel

Le benchmark CEC2017 provient du package officiel [tilleyd/cec2017-py](https://github.com/tilleyd/cec2017-py).

```bash
pip install git+https://github.com/tilleyd/cec2017-py.git
```

**Vérifier l'installation** :
```bash
python -c "from cec2017.functions import all_functions; print('CEC2017 OK:', len(all_functions), 'fonctions')"
```

Vous devriez voir : `CEC2017 OK: 30 fonctions`

### 6. Lancer l'application

```bash
streamlit run app.py
```

L'app s'ouvrira automatiquement à `http://localhost:8501`

---

## ⚠️ Dépannage

### Le message "Mode Fallback" apparaît

Si vous voyez `⚠️ Mode Fallback - Installer cec2017 pour résultats officiels`, cela signifie que le package CEC2017 n'est pas détecté dans votre environnement actuel.

**Solution** :

1. Vérifiez que vous êtes dans le bon environnement :
   ```bash
   conda activate cmaes
   ```

2. Réinstallez le package CEC2017 :
   ```bash
   pip install git+https://github.com/tilleyd/cec2017-py.git
   ```

3. Redémarrez l'application Streamlit

### Erreurs avec venv

Si vous avez des erreurs avec `python -m venv`, utilisez Conda à la place (voir instructions ci-dessus).

---

## Qu'est-ce que CMA-ES ?

**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) est un algorithme d'optimisation stochastique qui adapte sa stratégie en apprenant la géométrie du problème.

**Pourquoi ?**

- Robuste et sans paramètres sensibles
- Excellent pour optimisation continue en haute dimension
- État de l'art pour benchmark CEC2017


---

## Pages du Dashboard

### 1. Interactive Optimizer

Variez 6 paramètres et voyez l'effet en temps réel sur la convergence.

**Paramètres** :

- Taille population (10-100)
- σ initial (0.1-1.0)
- c_c, c_s, damping
- Max évaluations

### 2. Algorithm Explorer

Explications mathématiques et comportement de CMA-ES.

### 3. 3D Functions

Visualisez les 30 fonctions CEC2017 en 3D et 2D.

### 4. Algorithm Comparison

Comparez CMA-ES avec d'autres algorithmes d'optimisation.

### 5. Gbest vs Lbest

Comprenez les stratégies de partage d'information.

### 6. Benchmark Results ⭐

**Page principale pour le projet académique** :
- Exécutez le benchmark CEC2017 complet
- 30 runs par fonction (comme demandé)
- Tableau moyenne/écart-type
- Courbes de convergence pour F2, F4, F12, F25
- Export CSV et LaTeX

---

## Structure du Projet

```plaintext
cmaes-optimizer/
├── app.py                          # Page d'accueil
├── config.py                       # Configuration
├── requirements.txt                # Dépendances
├── algorithm/cmaes.py              # Implémentation CMA-ES
├── benchmark/cec2017.py            # CEC2017 (30 fonctions)
└── pages/
    ├── 1_Interactive_Optimizer.py
    ├── 2_Algorithm_Explorer.py
    ├── 3_3D_Functions.py
    ├── 4_Algorithm_Comparison.py
    ├── 5_Gbest_vs_Lbest.py
    └── 6_Benchmark_Results.py      # ⭐ Benchmark officiel
```

---

## Dépendances Principales

Voir `requirements.txt` :

- **Streamlit** : Dashboard interactif
- **NumPy** : Calculs numériques
- **SciPy** : Algèbre linéaire
- **Plotly** : Visualisations interactives


---

## Concepts Clés

### Fitness

Valeur que retourne la fonction à évaluer. **À minimiser**.

### Convergence

Le processus par lequel l'algorithme trouve l'optimum.

### CEC2017

Benchmark avec 30 fonctions de test :

- **F1-F3** : Unimodales (faciles)
- **F4-F10** : Multimodales (moyen)
- **F11-F20** : Hybrides (difficiles)
- **F21-F30** : Composées (très difficiles)



Projet académique - 2025
