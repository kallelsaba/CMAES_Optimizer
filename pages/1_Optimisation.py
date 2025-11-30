import streamlit as st
import numpy as np
import plotly.graph_objects as go
from algorithm.cmaes import CMAES
from benchmark.cec2017 import CEC2017

st.set_page_config(page_title="Interactive Optimizer", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    .slider-title {
        color: #4F46E5;
        font-weight: bold;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Interactive Optimizer")
st.write("Variez les paramètres et voyez l'effet en temps réel")

st.divider()

with st.sidebar:
    st.header("🎛️ Paramètres CMA-ES")
    
    func_id = st.slider("Fonction CEC2017", 1, 30, 5, help="Choisir une fonction de test")
    
    st.subheader("Population et Convergence")
    pop_size = st.slider("Taille population (λ)", 10, 100, 30, step=5,
                         help="Formule standard: λ = 4 + floor(3·ln(n)) ≈ 14 pour n=30")
    max_evals = st.slider("Max évaluations", 5000, 50000, 30000, step=5000)
    
    st.subheader("Adaptation Covariance")
    # σ initial en fraction du domaine (1/3 = optimal)
    sigma_fraction = st.slider("σ initial (×range)", 0.1, 0.5, 0.33, step=0.01,
                               help="Fraction du domaine. Optimal ≈ 1/3 = 0.33")
    sigma_init = sigma_fraction * 200  # Pour bounds [-100, 100], range = 200
    
    c_c = st.slider("c_c (covariance path)", 0.05, 0.5, 0.16, step=0.01,
                    help="Formule: (4 + μeff/n) / (n + 4 + 2·μeff/n)")
    
    st.subheader("Step-size Control")
    c_s = st.slider("c_s (step-size path)", 0.05, 0.5, 0.15, step=0.01,
                    help="Formule: (μeff + 2) / (n + μeff + 5)")
    damps = st.slider("d_σ (damping)", 0.5, 3.0, 1.15, step=0.05,
                      help="Formule: 1 + 2·max(0, √((μeff-1)/(n+1)) - 1) + c_s")
    
    st.divider()
    
    if st.button("🚀 Lancer Optimisation", use_container_width=True):
        st.session_state.run_optimization = True
    
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.run_optimization = False
        st.session_state.clear()

if st.session_state.get('run_optimization', False):
    st.info("Optimisation en cours...")
    
    f = CEC2017(func_id)
    
    algo = CMAES(dim=30, bounds=[-100, 100], max_eval=max_evals, seed=42)
    algo.sigma = sigma_init
    algo.cc = c_c
    algo.cs = c_s
    algo.damps = damps
    
    # <CHANGE> Adapter population size et recalculer les poids
    algo.pop_size = int(pop_size)
    algo.mu = algo.pop_size // 2
    
    # Recalculer les poids quand mu change
    weights = np.log(algo.mu + 0.5) - np.log(np.arange(1, algo.mu + 1))
    algo.weights = weights / np.sum(weights)
    algo.mueff = 1.0 / np.sum(algo.weights ** 2)
    
    progress_bar = st.progress(0)
    best_fitness_placeholder = st.empty()
    
    while algo.evals < max_evals:
        solutions = algo.ask()
        fitness = np.array([f(x) for x in solutions])
        algo.tell(solutions, fitness)
        algo.evals += len(solutions)
        
        progress = algo.evals / max_evals
        progress_bar.progress(min(progress, 1.0))
        
        best_fitness_placeholder.metric(
            "Meilleure fitness",
            f"{algo.best_fitness:.2e}",
            delta=f"{-np.diff(algo.history[-10:]).mean():.2e}" if len(algo.history) > 10 else None
        )
    
    st.success("Optimisation terminée !")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fitness finale", f"{algo.best_fitness:.2e}")
    
    with col2:
        st.metric("Évaluations totales", algo.evals)
    
    with col3:
        improvement = (algo.history[0] - algo.history[-1]) / algo.history[0] * 100
        st.metric("Amélioration", f"{improvement:.1f}%")
    
    st.divider()
    
    st.subheader("📈 Courbe de Convergence")
    
    fig = go.Figure()
    
    history = np.array(algo.history)
    evals = np.arange(len(history))
    
    fig.add_trace(go.Scatter(
        x=evals, y=history,
        mode='lines',
        name='Fitness',
        line=dict(
            color='#4F46E5',
            width=3,
            shape='spline'
        ),
        fill='tozeroy',
        fillcolor='rgba(79, 70, 229, 0.15)'
    ))
    
    fig.update_layout(
        title=f"Convergence - F{func_id}",
        xaxis_title="Itérations",
        yaxis_title="Fitness (log scale)",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white'
    )
    fig.update_yaxes(type="log")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Statistiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Fitness initiale** : {history[0]:.2e}")
        st.write(f"**Fitness finale** : {history[-1]:.2e}")
        st.write(f"**Réduction** : {(history[0] - history[-1]):.2e}")
    
    with col2:
        st.write(f"**Moyenne des 100 dernières** : {history[-100:].mean():.2e}")
        st.write(f"**Écart-type** : {history[-100:].std():.2e}")
        rate = np.polyfit(np.arange(len(history)-100, len(history)), history[-100:], 1)[0]
        st.write(f"**Taux de convergence** : {rate:.2e}/itération")

else:
    st.info("Cliquez sur '🚀 Lancer Optimisation' pour démarrer")