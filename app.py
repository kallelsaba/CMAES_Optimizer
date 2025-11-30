import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="CMA-ES Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    
    .main-title {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5em;
        font-weight: 900;
        text-align: center;
        margin: 30px 0;
    }
    
    .subtitle {
        color: #64748B;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    .card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(124, 58, 237, 0.05));
        border: 2px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        color: #1E293B;
    }
    
    .card h3 {
        color: #4F46E5;
    }
    
    .algo-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.05), rgba(16, 185, 129, 0.05));
        border: 2px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        color: #1E293B;
    }
    
    .math-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(37, 99, 235, 0.05));
        border: 2px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">CMA-ES Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive Optimization Dashboard with Real-time Visualization</div>', unsafe_allow_html=True)

st.divider()

# ============================================================================
# FONCTIONNALITÉS DU DASHBOARD
# ============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>⚙️ Contrôle Interactif</h3>
    <p>Ajustez les paramètres de CMA-ES en temps réel avec des curseurs</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>📊 Suivi Live</h3>
    <p>Visualisez la convergence de l'algorithme en direct</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>🎯 Visualisation 3D</h3>
    <p>Explorez les fonctions CEC2017 en trois dimensions</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# QU'EST-CE QUE CMA-ES ?
# ============================================================================
st.header("🧬 Qu'est-ce que CMA-ES ?")

st.markdown("""
**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) est un algorithme 
d'optimisation stochastique qui adapte une distribution gaussienne multidimensionnelle 
pour trouver l'optimum global.

C'est l'un des **meilleurs algorithmes d'optimisation continue** et l'état de l'art 
pour les problèmes en boîte noire (sans gradient).
""")

col_def1, col_def2 = st.columns(2)

with col_def1:
    st.markdown("""
    <div class="algo-card">
    <h4>🔑 Trois composantes clés</h4>
    <ul>
        <li><strong>m (Moyenne)</strong> : Centre de la distribution de recherche</li>
        <li><strong>C (Covariance)</strong> : Forme de l'ellipse de recherche</li>
        <li><strong>σ (Step-size)</strong> : Taille globale du pas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_def2:
    st.markdown("""
    <div class="card">
    <h4>✅ Avantages</h4>
    <ul>
        <li>Auto-adaptatif (pas de tuning manuel)</li>
        <li>Performant en haute dimension</li>
        <li>État de l'art sur CEC2017</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# ÉQUATIONS CLÉS
# ============================================================================
st.header("📐 Équations Clés")

# Utiliser un conteneur avec CSS Grid pour un alignement parfait
st.markdown("""
<style>
.equation-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}
.equation-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(37, 99, 235, 0.05));
    border: 2px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 15px 20px;
    min-height: 120px;
}
.equation-box h4 {
    color: #3B82F6;
    margin-bottom: 10px;
    font-size: 1em;
}
</style>
""", unsafe_allow_html=True)

# Première ligne
row1 = st.columns(2)

with row1[0]:
    st.markdown('<div class="equation-box"><h4>📊 Génération de population</h4></div>', unsafe_allow_html=True)
    st.latex(r"x_k \sim m + \sigma \cdot \mathcal{N}(0, C)")

with row1[1]:
    st.markdown('<div class="equation-box"><h4>📍 Mise à jour de la moyenne</h4></div>', unsafe_allow_html=True)
    st.latex(r"m \leftarrow \sum_{i=1}^{\mu} w_i \, x_{i:\lambda}")

# Deuxième ligne  
row2 = st.columns(2)

with row2[0]:
    st.markdown('<div class="equation-box"><h4>📏 Adaptation du step-size</h4></div>', unsafe_allow_html=True)
    st.latex(r"\sigma \leftarrow \sigma \cdot \exp\left(\frac{c_s}{d_s}\left(\frac{\|p_s\|}{\chi_n} - 1\right)\right)")

with row2[1]:
    st.markdown('<div class="equation-box"><h4>🔄 Adaptation de la covariance</h4></div>', unsafe_allow_html=True)
    st.latex(r"C \leftarrow (1-c_1-c_\mu)C + c_1 p_c p_c^T + c_\mu \sum w_i y_i y_i^T")

st.divider()

# ============================================================================
# EFFET DE SIGMA SUR LA CONVERGENCE
# ============================================================================
st.header("📈 Effet de σ initial sur la Convergence")

np.random.seed(42)  # Pour reproductibilité
iterations = np.arange(100)
fig = go.Figure()

# Simulation réaliste de l'effet de σ initial
# σ optimal ≈ 1/3 du domaine donne la meilleure convergence

# σ = 0.1×range : Trop petit - convergence très lente (sous-exploration)
convergence_small = 1000 * np.exp(-iterations * 0.015) + 50

# σ = 0.33×range : Optimal - meilleure convergence
convergence_optimal = 1000 * np.exp(-iterations * 0.06) + 0.01

# σ = 0.5×range : Un peu grand - convergence ok mais moins précise
convergence_medium = 1000 * np.exp(-iterations * 0.04) + 2

# σ = 1.0×range : Trop grand - instable, oscillations, mauvaise convergence
base = 1000 * np.exp(-iterations * 0.02)
noise = 30 * np.sin(iterations * 0.3) * np.exp(-iterations * 0.01)
convergence_large = base + 20 + np.abs(noise)

data = [
    (convergence_small, '#EF4444', 'σ = 0.1×range (trop petit)'),
    (convergence_optimal, '#22C55E', 'σ = 0.33×range (optimal ✓)'),
    (convergence_medium, '#3B82F6', 'σ = 0.5×range (acceptable)'),
    (convergence_large, '#F97316', 'σ = 1.0×range (trop grand)')
]

for convergence, color, label in data:
    fig.add_trace(go.Scatter(
        x=iterations, y=convergence,
        name=label,
        line=dict(color=color, width=3)
    ))

fig.update_layout(
    xaxis_title="Itérations",
    yaxis_title="Fitness (échelle log)",
    template="plotly_white",
    height=400,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)
fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Interprétation** : 
- **σ trop petit** (0.1×range) : Convergence très lente, exploration insuffisante
- **σ optimal** (≈0.33×range = 1/3) : Meilleur compromis exploration/exploitation  
- **σ trop grand** (1.0×range) : Instabilité, convergence erratique, peut diverger

CMA-ES adapte σ automatiquement via le mécanisme CSA (Cumulative Step-size Adaptation) !
""")

st.divider()

# ============================================================================
# PARAMÈTRES CLÉS
# ============================================================================
st.header("🎛️ Paramètres Clés")

col_p1, col_p2 = st.columns(2)

with col_p1:
    with st.expander("📌 σ (sigma) - Step-size"):
        st.write("Contrôle la portée de la recherche. Trop petit = convergence lente. Trop grand = instabilité.")
    
    with st.expander("📌 c_c - Coefficient évolution path"):
        st.write("Coefficient de mise à jour du chemin d'évolution pour la covariance. Valeurs typiques : [0.01, 0.5]")
    
    with st.expander("📌 c_s - Coefficient step-size"):
        st.write("Coefficient pour l'adaptation du step-size. Contrôle la vitesse d'ajustement de σ.")

with col_p2:
    with st.expander("📌 c_1 - Coefficient rang-1"):
        st.write("Coefficient rang-1 update. Utilise le chemin d'évolution pour mettre à jour C.")
    
    with st.expander("📌 c_μ - Coefficient rang-μ"):
        st.write("Coefficient rang-μ update. Utilise les μ meilleures solutions pour mettre à jour C.")
    
    with st.expander("📌 λ (lambda) - Taille population"):
        st.write("Nombre d'individus générés par génération. Plus grand = plus robuste mais plus lent.")

st.divider()

# ============================================================================
# NAVIGATION
# ============================================================================
st.header("📍 Navigation")

st.markdown("""
Sélectionnez une page dans la barre latérale gauche :

| Page | Description |
|------|-------------|
| **Interactive Optimizer** | Contrôlez les paramètres et voyez l'effet en temps réel |
| **3D Functions** | Visualisez les fonctions CEC2017 en 3D |
| **Algorithm Comparison** | Comparez CMA-ES avec d'autres algorithmes |
| **Gbest vs Lbest** | Stratégies de partage d'information |
| **Benchmark Results** | Exécutez le benchmark officiel (30 runs, tableau résultats) |
""")

st.divider()

# ============================================================================
# RÉFÉRENCES
# ============================================================================
st.markdown("""
**Références** :
- Hansen & Ostermeier (2003) - *"Completely Derandomized Self-Adaptation in Evolution Strategies"*
- CMA-ES est utilisé en robotique, apprentissage par renforcement, et optimisation de hyperparamètres

---
**Projet Académique** - Méthodes Heuristiques et Métaheuristiques - 2025
""")