import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Global Best vs Local Best", 
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    .gbest-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .lbest-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.1));
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .cmaes-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(79, 70, 229, 0.1));
        border: 2px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Global Best vs Local Best vs CMA-ES")
st.markdown("### Comment CMA-ES transcende le dilemme gbest/lbest")
st.divider()

st.header("Concepts Fondamentaux")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="gbest-card">
    <h3>Global Best (gbest)</h3>
    <p>Le <strong>meilleur individu</strong> de toute la population.</p>
    <h4>Principe</h4>
    <ul>
        <li>Tous les agents connaissent la meilleure solution globale</li>
        <li>Communication totale dans la population</li>
    </ul>
    <h4>Caracteristiques</h4>
    <ul>
        <li>Convergence rapide</li>
        <li>Efficace sur fonctions unimodales</li>
        <li>Convergence prematuree possible</li>
        <li>Risque de piege dans minima locaux</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="lbest-card">
    <h3>Local Best (lbest)</h3>
    <p>Le <strong>meilleur individu</strong> dans un voisinage local.</p>
    <h4>Principe</h4>
    <ul>
        <li>Chaque agent a un voisinage limite</li>
        <li>Communication locale uniquement</li>
    </ul>
    <h4>Caracteristiques</h4>
    <ul>
        <li>Exploration diversifiee</li>
        <li>Evite les minima locaux</li>
        <li>Convergence plus lente</li>
        <li>Peut manquer l optimum global</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Visualisation des Topologies")

col1, col2 = st.columns(2)

n_agents = 10
angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
x_agents = np.cos(angles)
y_agents = np.sin(angles)

with col1:
    fig_gbest = go.Figure()
    fig_gbest.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
        marker=dict(size=30, color="#22C55E", symbol="star"), name="gbest"))
    fig_gbest.add_trace(go.Scatter(x=x_agents, y=y_agents, mode="markers",
        marker=dict(size=15, color="#3B82F6"), name="Agents"))
    for i in range(n_agents):
        fig_gbest.add_trace(go.Scatter(x=[x_agents[i], 0], y=[y_agents[i], 0],
            mode="lines", line=dict(color="rgba(34, 197, 94, 0.3)", width=2),
            showlegend=False, hoverinfo="skip"))
    fig_gbest.update_layout(title="Topologie Global Best",
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white", height=350)
    st.plotly_chart(fig_gbest, use_container_width=True)
    st.caption("Tous les agents sont connectes au meilleur global")

with col2:
    fig_lbest = go.Figure()
    fig_lbest.add_trace(go.Scatter(x=x_agents, y=y_agents, mode="markers",
        marker=dict(size=15, color="#3B82F6"), name="Agents"))
    for i in range(n_agents):
        next_i = (i + 1) % n_agents
        fig_lbest.add_trace(go.Scatter(x=[x_agents[i], x_agents[next_i]], 
            y=[y_agents[i], y_agents[next_i]], mode="lines",
            line=dict(color="rgba(59, 130, 246, 0.3)", width=2),
            showlegend=False, hoverinfo="skip"))
    lbest_indices = [0, 3, 7]
    fig_lbest.add_trace(go.Scatter(x=x_agents[lbest_indices], y=y_agents[lbest_indices],
        mode="markers", marker=dict(size=25, color="#F59E0B", symbol="star"), name="lbest locaux"))
    fig_lbest.update_layout(title="Topologie Local Best",
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white", height=350)
    st.plotly_chart(fig_lbest, use_container_width=True)
    st.caption("Chaque agent communique avec ses voisins proches")

st.info("""
**Exploration vs Exploitation** : Le choix gbest/lbest est un compromis fondamental.
- **gbest** = Plus d exploitation (convergence rapide)
- **lbest** = Plus d exploration (recherche diversifiee)
""")

st.divider()

st.header("CMA-ES : Un Paradigme Superieur")

st.markdown("""
**CMA-ES n utilise ni gbest ni lbest**, mais une approche plus sophistiquee qui combine 
les avantages des deux sans leurs inconvenients.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="lbest-card">
    <h3>Approche classique (gbest/lbest)</h3>
    <ul>
        <li>Agents individuels avec memoire</li>
        <li>Communication via le meilleur</li>
        <li>Parametres fixes</li>
        <li>Mouvement spherique uniforme</li>
    </ul>
    <h4>Limitation</h4>
    <p>L information d un seul meilleur peut etre trompeuse.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="cmaes-card">
    <h3>Approche CMA-ES</h3>
    <ul>
        <li>Distribution gaussienne adaptative</li>
        <li>Utilise les <strong>mu meilleurs</strong> (pas un seul)</li>
        <li>Auto-adaptation de sigma et C</li>
        <li>Mouvement ellipsoidal oriente</li>
    </ul>
    <h4>Avantage cle</h4>
    <p>Moyenne ponderee + adaptation geometrique = robustesse.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Les 3 Mecanismes Cles de CMA-ES")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Moyenne Ponderee
    
    CMA-ES utilise les **mu meilleurs** individus avec des poids decroissants.
    
    - Plus robuste qu un seul gbest
    - Une mauvaise solution n a pas d impact drastique
    """)

with col2:
    st.markdown("""
    ### 2. Matrice de Covariance C
    
    La matrice C capture les **correlations** entre variables.
    
    - Equivalent a un lbest adaptatif
    - S oriente vers les directions prometteuses
    """)

with col3:
    st.markdown("""
    ### 3. Step-size sigma
    
    Le step-size controle l exploration/exploitation **automatiquement**.
    
    - Grand sigma = exploration (comme lbest)
    - Petit sigma = exploitation (comme gbest)
    """)

st.divider()

st.subheader("Comparaison Visuelle des Distributions")

fig_comp = make_subplots(rows=1, cols=3,
    subplot_titles=("gbest : Convergence unique", "lbest : Clusters multiples", "CMA-ES : Ellipse adaptative"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]])

np.random.seed(42)

x_pso_gbest = np.random.randn(50) * 0.3 + 2
y_pso_gbest = np.random.randn(50) * 0.3 + 2
fig_comp.add_trace(go.Scatter(x=x_pso_gbest, y=y_pso_gbest, mode="markers", 
    marker=dict(color="#22C55E", size=8), showlegend=False), row=1, col=1)
fig_comp.add_trace(go.Scatter(x=[2], y=[2], mode="markers", 
    marker=dict(color="red", size=20, symbol="star"), showlegend=False), row=1, col=1)

x_lbest_1 = np.random.randn(15) * 0.4 + 1
y_lbest_1 = np.random.randn(15) * 0.4 + 1
x_lbest_2 = np.random.randn(15) * 0.4 + 3
y_lbest_2 = np.random.randn(15) * 0.4 + 3
x_lbest_3 = np.random.randn(15) * 0.4 + 2
y_lbest_3 = np.random.randn(15) * 0.4 + 0.5
fig_comp.add_trace(go.Scatter(x=np.concatenate([x_lbest_1, x_lbest_2, x_lbest_3]), 
    y=np.concatenate([y_lbest_1, y_lbest_2, y_lbest_3]), 
    mode="markers", marker=dict(color="#3B82F6", size=8), showlegend=False), row=1, col=2)

theta = np.linspace(0, 2*np.pi, 100)
angle = np.pi/6
a, b = 1.5, 0.7
x_ellipse = 2 + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
y_ellipse = 2 + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
n_samples = 50
t_samples = np.random.rand(n_samples) * 2 * np.pi
r_samples = np.random.randn(n_samples) * 0.3 + 1
x_cmaes = 2 + r_samples * (a * np.cos(t_samples) * np.cos(angle) - b * np.sin(t_samples) * np.sin(angle))
y_cmaes = 2 + r_samples * (a * np.cos(t_samples) * np.sin(angle) + b * np.sin(t_samples) * np.cos(angle))
fig_comp.add_trace(go.Scatter(x=x_cmaes, y=y_cmaes, mode="markers", 
    marker=dict(color="#7C3AED", size=8), showlegend=False), row=1, col=3)
fig_comp.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode="lines", 
    line=dict(color="#7C3AED", width=3), showlegend=False), row=1, col=3)

fig_comp.update_xaxes(range=[0, 4], showgrid=True)
fig_comp.update_yaxes(range=[0, 4], showgrid=True)
fig_comp.update_layout(height=400, template="plotly_white")

st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("""
| Aspect | gbest | lbest | CMA-ES |
|--------|-------|-------|--------|
| **Information** | 1 seul individu | Voisinage local | mu meilleurs ponderes |
| **Adaptation** | Parametres fixes | Parametres fixes | Auto-adaptation (C, sigma) |
| **Geometrie** | Spherique | Spherique | Ellipsoide adaptatif |
| **Convergence** | Lineaire | Lineaire | Superlineaire |
""")

st.divider()

# ============================================================================
# VISUALISATION INTERACTIVE
# ============================================================================
st.header("🎮 Visualisation Interactive")

st.markdown("""
Explorez comment **CMA-ES** adapte sa distribution au fil des iterations, 
comparé aux approches gbest et lbest.
""")

col_params1, col_params2, col_params3 = st.columns(3)

with col_params1:
    algo_choice = st.selectbox("Algorithme", ["CMA-ES", "PSO-gbest", "PSO-lbest"])

with col_params2:
    n_particles = st.slider("Nombre d'agents (λ)", 10, 100, 30)

with col_params3:
    iteration = st.slider("Iteration", 0, 50, 0)

# Fonction Rastrigin 2D pour le fond
x_grid = np.linspace(-5, 5, 100)
y_grid = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = 20 + X_grid**2 + Y_grid**2 - 10*(np.cos(2*np.pi*X_grid) + np.cos(2*np.pi*Y_grid))

# Optimum global
opt_x, opt_y = 0, 0

# Simuler l'evolution selon l'algorithme
np.random.seed(42)

if algo_choice == "CMA-ES":
    # CMA-ES: l'ellipse se deplace et se contracte vers l'optimum
    # Position moyenne qui converge vers l'optimum
    progress = min(iteration / 50, 1.0)
    mean_x = 3 * (1 - progress) + opt_x * progress
    mean_y = 2 * (1 - progress) + opt_y * progress
    
    # Sigma diminue (exploitation croissante)
    sigma = 2.0 * (1 - 0.8 * progress)
    
    # L'ellipse s'oriente vers l'optimum (rotation adaptative)
    angle = -np.pi/4 * progress
    
    # Axes de l'ellipse (adaptation de la covariance)
    a = sigma * (1.5 - 0.5 * progress)  # axe principal
    b = sigma * (0.8 - 0.3 * progress)  # axe secondaire
    
    # Generer les points selon la distribution gaussienne elliptique
    theta_samples = np.random.rand(n_particles) * 2 * np.pi
    r_samples = np.abs(np.random.randn(n_particles))
    
    x_agents = mean_x + r_samples * (a * np.cos(theta_samples) * np.cos(angle) - b * np.sin(theta_samples) * np.sin(angle))
    y_agents = mean_y + r_samples * (a * np.cos(theta_samples) * np.sin(angle) + b * np.sin(theta_samples) * np.cos(angle))
    
    # Ellipse de confiance (1-sigma)
    theta_ellipse = np.linspace(0, 2*np.pi, 100)
    x_ellipse = mean_x + a * np.cos(theta_ellipse) * np.cos(angle) - b * np.sin(theta_ellipse) * np.sin(angle)
    y_ellipse = mean_y + a * np.cos(theta_ellipse) * np.sin(angle) + b * np.sin(theta_ellipse) * np.cos(angle)
    
    best_x, best_y = mean_x, mean_y
    algo_color = "#7C3AED"
    
elif algo_choice == "PSO-gbest":
    # PSO-gbest: tous convergent vers le meme point (potentiellement un minimum local)
    progress = min(iteration / 50, 1.0)
    
    # Le gbest peut etre piege dans un minimum local (-2, 2)
    if iteration < 20:
        gbest_x = -2 + 0.05 * iteration
        gbest_y = 2 - 0.05 * iteration
    else:
        gbest_x = -1
        gbest_y = 1
    
    # Tous les agents convergent vers gbest
    spread = max(3 * (1 - progress), 0.3)
    x_agents = np.random.randn(n_particles) * spread + gbest_x
    y_agents = np.random.randn(n_particles) * spread + gbest_y
    
    best_x, best_y = gbest_x, gbest_y
    x_ellipse, y_ellipse = None, None
    algo_color = "#22C55E"
    
else:  # PSO-lbest
    # PSO-lbest: plusieurs clusters qui explorent independamment
    progress = min(iteration / 50, 1.0)
    
    # 3 clusters avec leurs propres lbest
    n_per_cluster = n_particles // 3
    spread = max(2 * (1 - 0.5 * progress), 0.5)
    
    # Cluster 1: explore vers (-2, 2)
    lbest1_x = -2 + 0.5 * progress
    lbest1_y = 2 - 0.5 * progress
    
    # Cluster 2: explore vers (2, 2)  
    lbest2_x = 2 - 0.8 * progress
    lbest2_y = 2 - 1.0 * progress
    
    # Cluster 3: explore vers (0, -2) et trouve presque l'optimum!
    lbest3_x = 0
    lbest3_y = -2 + 1.5 * progress
    
    x_agents = np.concatenate([
        np.random.randn(n_per_cluster) * spread + lbest1_x,
        np.random.randn(n_per_cluster) * spread + lbest2_x,
        np.random.randn(n_particles - 2*n_per_cluster) * spread + lbest3_x
    ])
    y_agents = np.concatenate([
        np.random.randn(n_per_cluster) * spread + lbest1_y,
        np.random.randn(n_per_cluster) * spread + lbest2_y,
        np.random.randn(n_particles - 2*n_per_cluster) * spread + lbest3_y
    ])
    
    best_x, best_y = lbest3_x, lbest3_y  # Le meilleur lbest
    x_ellipse, y_ellipse = None, None
    algo_color = "#3B82F6"

# Creer la figure
fig_interactive = go.Figure()

# Contour de la fonction (Rastrigin)
fig_interactive.add_trace(go.Contour(
    x=x_grid, y=y_grid, z=Z_grid,
    colorscale='Viridis',
    opacity=0.5,
    showscale=False,
    contours=dict(showlabels=False),
    hoverinfo='skip'
))

# Agents
fig_interactive.add_trace(go.Scatter(
    x=x_agents, y=y_agents,
    mode='markers',
    marker=dict(size=10, color=algo_color, line=dict(color='white', width=1)),
    name='Agents'
))

# Ellipse CMA-ES
if algo_choice == "CMA-ES" and x_ellipse is not None:
    fig_interactive.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        line=dict(color=algo_color, width=3, dash='dash'),
        name='Distribution (1σ)'
    ))

# Marquer le meilleur/moyenne
fig_interactive.add_trace(go.Scatter(
    x=[best_x], y=[best_y],
    mode='markers',
    marker=dict(size=20, color='#EF4444', symbol='star', line=dict(color='white', width=2)),
    name='Best/Mean' if algo_choice == "CMA-ES" else 'gbest' if algo_choice == "PSO-gbest" else 'Meilleur lbest'
))

# Optimum global
fig_interactive.add_trace(go.Scatter(
    x=[opt_x], y=[opt_y],
    mode='markers',
    marker=dict(size=25, color='#10B981', symbol='x', line=dict(color='white', width=3)),
    name='Optimum Global'
))

# Pour lbest, marquer les autres lbest
if algo_choice == "PSO-lbest":
    fig_interactive.add_trace(go.Scatter(
        x=[lbest1_x, lbest2_x], y=[lbest1_y, lbest2_y],
        mode='markers',
        marker=dict(size=15, color='#F59E0B', symbol='star'),
        name='Autres lbest'
    ))

fig_interactive.update_layout(
    title=f"{algo_choice} - Iteration {iteration}",
    xaxis_title="x₁",
    yaxis_title="x₂",
    template="plotly_white",
    height=550,
    xaxis=dict(range=[-5, 5]),
    yaxis=dict(range=[-5, 5]),
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig_interactive, use_container_width=True)

# Metrics
col_m1, col_m2, col_m3 = st.columns(3)

distance_to_opt = np.sqrt((best_x - opt_x)**2 + (best_y - opt_y)**2)

with col_m1:
    st.metric("Distance à l'optimum", f"{distance_to_opt:.3f}")

with col_m2:
    if algo_choice == "CMA-ES":
        st.metric("Step-size (σ)", f"{sigma:.3f}")
    else:
        st.metric("Dispersion", f"{spread:.3f}")

with col_m3:
    if algo_choice == "CMA-ES":
        st.metric("Position moyenne", f"({mean_x:.2f}, {mean_y:.2f})")
    elif algo_choice == "PSO-gbest":
        st.metric("Position gbest", f"({gbest_x:.2f}, {gbest_y:.2f})")
    else:
        st.metric("Meilleur lbest", f"({lbest3_x:.2f}, {lbest3_y:.2f})")

# Explication selon l'algorithme
if algo_choice == "CMA-ES":
    st.success(f"""
    **CMA-ES (Iteration {iteration})** : 
    - L'ellipse (distribution gaussienne) se **deplace** vers l'optimum
    - Elle **se contracte** (σ diminue) pour affiner la recherche
    - Elle **s'oriente** vers les directions prometteuses (adaptation de C)
    - Utilise les **μ meilleurs** pour mettre a jour la moyenne
    """)
elif algo_choice == "PSO-gbest":
    if iteration < 20:
        st.warning(f"""
        **PSO-gbest (Iteration {iteration})** : 
        - Tous les agents convergent vers le **meme gbest**
        - Convergence rapide mais... le gbest est piege dans un **minimum local** !
        - Une fois la diversite perdue, impossible d'en sortir
        """)
    else:
        st.error(f"""
        **PSO-gbest (Iteration {iteration})** : 
        - **Convergence prematuree** : tous pieges autour de (-1, 1)
        - L'optimum global (0, 0) ne sera jamais atteint
        - C'est le probleme classique de gbest sur fonctions multimodales
        """)
else:
    st.info(f"""
    **PSO-lbest (Iteration {iteration})** : 
    - **3 clusters** explorent independamment avec leurs propres lbest
    - Le cluster du bas (lbest3) s'approche de l'optimum global
    - Convergence plus lente mais **plus robuste**
    - Meilleure exploration de l'espace de recherche
    """)

st.divider()

st.markdown("""
### Conclusion

Le dilemme **gbest vs lbest** est un compromis classique en optimisation :
- **gbest** = convergence rapide mais risque de piege
- **lbest** = exploration robuste mais lente

**CMA-ES transcende ce dilemme** grace a :
1. **Moyenne ponderee** des mu meilleurs (pas un seul gbest)
2. **Adaptation geometrique** via la matrice de covariance C
3. **Controle automatique** exploration/exploitation via sigma

C est pourquoi **CMA-ES est l etat de l art** pour l optimisation continue en boite noire !
""")
