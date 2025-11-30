import streamlit as st
import numpy as np
import plotly.graph_objects as go
from benchmark.cec2017 import CEC2017, CEC2017_OFFICIAL

# Importer les fonctions CEC2017 officielles
# NOTE: Seules F1-F10 et F21-F28 supportent la dimension 2
# F11-F20 (hybrides) et F29-F30 ne supportent pas la dimension 2
if CEC2017_OFFICIAL:
    try:
        from cec2017.functions import (
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f21, f22, f23, f24, f25, f26, f27, f28
        )
        HAS_CEC = True
        
        # Fonctions supportant la dimension 2 (comme indiqué dans le README)
        # F1-F10: Unimodales et Multimodales
        # F21-F28: Composées (supportent dim 2)
        # F11-F20: Hybrides (NE supportent PAS dim 2)
        # F29-F30: Composées (NE supportent PAS dim 2)
        CEC_FUNCTIONS_2D = {
            1: ("Shifted and Rotated Bent Cigar", f1),
            2: ("Shifted and Rotated Sum of Different Power (DEPRECATED)", f2),
            3: ("Shifted and Rotated Zakharov", f3),
            4: ("Shifted and Rotated Rosenbrock", f4),
            5: ("Shifted and Rotated Rastrigin", f5),
            6: ("Shifted and Rotated Expanded Scaffer's F6", f6),
            7: ("Shifted and Rotated Lunacek Bi-Rastrigin", f7),
            8: ("Shifted and Rotated Non-Continuous Rastrigin", f8),
            9: ("Shifted and Rotated Levy", f9),
            10: ("Shifted and Rotated Schwefel", f10),
            # F11-F20 ne supportent pas dim 2 (hybrides avec shuffle)
            21: ("Composition Function 1", f21),
            22: ("Composition Function 2", f22),
            23: ("Composition Function 3", f23),
            24: ("Composition Function 4", f24),
            25: ("Composition Function 5", f25),
            26: ("Composition Function 6", f26),
            27: ("Composition Function 7", f27),
            28: ("Composition Function 8", f28),
            # F29-F30 ne supportent pas dim 2 (composition avec shuffle)
        }
        
        # Fonctions disponibles pour visualisation 2D
        AVAILABLE_FUNCS = list(CEC_FUNCTIONS_2D.keys())
        
    except ImportError:
        HAS_CEC = False
else:
    HAS_CEC = False

st.set_page_config(
    page_title="3D Functions", 
    layout="wide",
)

# CSS mode clair
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 3D CEC2017 Functions Visualization")

# Afficher le statut
if CEC2017_OFFICIAL and HAS_CEC:
    st.success("✅ Benchmark CEC2017 officiel actif")
else:
    st.warning("⚠️ Mode fallback - Installer cec2017 pour visualisations officielles")

st.write("""
Explorez les fonctions CEC2017 en 3D. 
**Note**: Seules F1-F10 et F21-F28 supportent la dimension 2 pour la visualisation.
F11-F20 (hybrides) et F29-F30 nécessitent une dimension ≥ 10.
""")

st.divider()

# Sélection fonction
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if HAS_CEC:
        func_id = st.selectbox(
            "Choisir une fonction CEC2017",
            options=AVAILABLE_FUNCS,
            format_func=lambda x: f"F{x} - {CEC_FUNCTIONS_2D[x][0]}"
        )
    else:
        func_id = st.slider("Choisir une fonction CEC2017", 1, 30, 1)

with col2:
    resolution = st.select_slider("Résolution", options=[30, 50, 80, 100], value=50)

with col3:
    use_log_scale = st.checkbox("Échelle log", value=False)

# Créer la grille 2D pour visualisation
x_range = np.linspace(-100, 100, resolution)
y_range = np.linspace(-100, 100, resolution)
X, Y = np.meshgrid(x_range, y_range)

# Calculer Z pour chaque point
Z = np.zeros_like(X)

# Créer tous les points en une seule matrice pour efficacité
points = np.column_stack([X.ravel(), Y.ravel()])

if HAS_CEC and func_id in CEC_FUNCTIONS_2D:
    # Utiliser les fonctions CEC2017 officielles
    func_name, func = CEC_FUNCTIONS_2D[func_id]
    Z_flat = func(points)
    Z = Z_flat.reshape(X.shape)
else:
    # Fallback: utiliser notre implémentation
    f = CEC2017(func_id, dim=2)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    func_name = f"F{func_id} (fallback)"

# Appliquer l'échelle log si demandé
if use_log_scale:
    Z_display = np.log10(np.abs(Z) + 1)
    z_label = "log₁₀(|f(x)| + 1)"
else:
    Z_display = Z
    z_label = "f(x)"

# Créer la surface 3D
fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z_display,
    colorscale='Viridis',
    colorbar=dict(title=z_label),
    lighting=dict(
        ambient=0.6,
        diffuse=0.8,
        specular=0.2,
        roughness=0.5
    )
)])

# Déterminer la catégorie de fonction
if func_id <= 3:
    category = "Unimodale"
elif func_id <= 10:
    category = "Multimodale"
elif func_id <= 20:
    category = "Hybride"
else:
    category = "Composée"

# Titre avec le nom de la fonction
if HAS_CEC and func_id in CEC_FUNCTIONS_2D:
    title = f"F{func_id} - {func_name}"
else:
    title = f"CEC2017 F{func_id} ({category})"

fig.update_layout(
    title=title,
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title=z_label,
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        bgcolor='rgba(248, 250, 252, 0.8)'
    ),
    template="plotly_white",
    height=700,
    hovermode='closest',
    paper_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Informations sur la fonction
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 Statistiques F{func_id}")
    st.write(f"**Catégorie**: {category}")
    st.write(f"**Min observé**: {Z.min():.2e}")
    st.write(f"**Max observé**: {Z.max():.2e}")
    st.write(f"**Moyenne**: {Z.mean():.2e}")

with col2:
    st.subheader("📖 Description")
    descriptions = {
        (1, 3): "Fonctions unimodales avec un seul optimum global. Relativement faciles à optimiser.",
        (4, 10): "Fonctions multimodales avec plusieurs optima locaux. Plus difficiles car l'algorithme peut être piégé.",
        (11, 20): "Fonctions hybrides combinant plusieurs fonctions de base. Très challenging.",
        (21, 30): "Fonctions composées avec structure complexe. Les plus difficiles du benchmark."
    }
    for (start, end), desc in descriptions.items():
        if start <= func_id <= end:
            st.write(desc)
            break

st.divider()

# Contour plot
st.subheader("🗺️ Vue de dessus (Contour)")

fig_contour = go.Figure(data=[go.Contour(
    x=x_range, 
    y=y_range, 
    z=Z_display,
    colorscale='Viridis',
    colorbar=dict(title=z_label),
    contours=dict(
        showlabels=True,
        labelfont=dict(size=10, color='white')
    )
)])

fig_contour.update_layout(
    title=f"Contour Plot - F{func_id}",
    xaxis_title="x",
    yaxis_title="y",
    template="plotly_white",
    height=500,
    hovermode='closest',
    plot_bgcolor='rgba(248, 250, 252, 0.5)',
    paper_bgcolor='white'
)

st.plotly_chart(fig_contour, use_container_width=True)

# Note sur la visualisation
st.info("""
**Note**: Cette visualisation montre la fonction en 2D. Le benchmark CEC2017 est normalement utilisé 
en dimension 30 pour les tests d'algorithmes. La complexité réelle des fonctions est bien plus élevée 
en haute dimension.
""")