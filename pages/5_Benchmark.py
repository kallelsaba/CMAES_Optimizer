import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from algorithm.cmaes import CMAES
from benchmark.cec2017 import CEC2017, CEC2017_OFFICIAL
import time

st.set_page_config(
    page_title="Benchmark Results",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    .benchmark-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(124, 58, 237, 0.05));
        border: 2px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .official-badge {
        background: linear-gradient(135deg, #22C55E, #16A34A);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .fallback-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Benchmark CEC2017 - Résultats Officiels")

# Afficher le statut du benchmark
if CEC2017_OFFICIAL:
    st.markdown('<span class="official-badge">✅ Benchmark CEC2017 Officiel Actif</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="fallback-badge">⚠️ Mode Fallback - Installer cec2017 pour résultats officiels</span>', unsafe_allow_html=True)
    st.code("pip install git+https://github.com/tilleyd/cec2017-py.git", language="bash")

st.markdown("""
### Exécution conforme aux requirements du projet
- **Dimension** : D = 30
- **Évaluations max** : 30000 (10000 × D)
- **Runs par fonction** : 30
- **Métriques** : Moyenne et Écart-type
""")

st.divider()

# Tabs
tabs = st.tabs([
    "🚀 Exécuter Benchmark",
    "📋 Tableau des Résultats",
    "📈 Courbes de Convergence",
    "📥 Exporter Résultats"
])

# ============================================================================
# TAB 1: EXÉCUTER BENCHMARK
# ============================================================================
with tabs[0]:
    st.header("🚀 Lancer le Benchmark CEC2017")
    
    st.markdown("""
    <div class="benchmark-card">
    <h4>⚠️ Configuration selon les requirements du projet</h4>
    <ul>
        <li><strong>30 exécutions</strong> par fonction pour la reproductibilité</li>
        <li><strong>30000 évaluations max</strong> (critère d'arrêt)</li>
        <li><strong>Dimension 30</strong> pour toutes les fonctions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sélection des fonctions")
        
        # Options de sélection
        selection_mode = st.radio(
            "Mode de sélection",
            ["Fonctions individuelles", "Par catégorie", "Toutes (F1-F30)"]
        )
        
        if selection_mode == "Fonctions individuelles":
            functions_to_test = st.multiselect(
                "Choisir les fonctions",
                options=list(range(1, 31)),
                default=[2, 4, 12, 25],  # Fonctions demandées pour les courbes
                format_func=lambda x: f"F{x}"
            )
        elif selection_mode == "Par catégorie":
            categories = st.multiselect(
                "Catégories",
                ["Unimodales (F1-F3)", "Multimodales (F4-F10)", "Hybrides (F11-F20)", "Composées (F21-F30)"],
                default=["Unimodales (F1-F3)"]
            )
            functions_to_test = []
            if "Unimodales (F1-F3)" in categories:
                functions_to_test.extend([1, 2, 3])
            if "Multimodales (F4-F10)" in categories:
                functions_to_test.extend([4, 5, 6, 7, 8, 9, 10])
            if "Hybrides (F11-F20)" in categories:
                functions_to_test.extend(list(range(11, 21)))
            if "Composées (F21-F30)" in categories:
                functions_to_test.extend(list(range(21, 31)))
        else:
            functions_to_test = list(range(1, 31))
    
    with col2:
        st.subheader("Paramètres d'exécution")
        
        num_runs = st.slider("Nombre de runs par fonction", 1, 30, 30, 
                            help="Le projet demande 30 runs")
        max_evals = st.number_input("Max évaluations", 1000, 100000, 30000, 
                                    help="Le projet demande 30000")
        
        # Paramètres CMA-ES optionnels
        with st.expander("⚙️ Paramètres CMA-ES avancés"):
            sigma_init = st.slider("σ initial", 10.0, 100.0, 66.67, step=1.0,
                                   help="Valeur standard = range/3 = 200/3 ≈ 66.67")
            pop_multiplier = st.slider("Multiplicateur population", 1.0, 3.0, 1.0, step=0.5,
                                       help="Multiplie λ par ce facteur (λ standard ≈ 14 pour n=30)")
    
    st.divider()
    
    # Bouton de lancement
    if st.button("🚀 Lancer le Benchmark Complet", use_container_width=True, type="primary"):
        if not functions_to_test:
            st.error("Veuillez sélectionner au moins une fonction")
        else:
            st.session_state.benchmark_running = True
            st.session_state.benchmark_results = {}
            st.session_state.convergence_curves = {}
            
            total_runs = len(functions_to_test) * num_runs
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_placeholder = st.empty()
            
            all_results = {}
            all_curves = {}
            
            current_run = 0
            start_time = time.time()
            
            for fid in functions_to_test:
                f = CEC2017(fid)
                function_results = []
                function_curves = []
                
                for run in range(num_runs):
                    current_run += 1
                    
                    # Créer algorithme avec seed différent pour chaque run
                    algo = CMAES(dim=30, bounds=[-100, 100], max_eval=max_evals, seed=run)
                    algo.sigma = sigma_init
                    
                    # Ajuster population si demandé
                    if pop_multiplier != 1.0:
                        algo.pop_size = int(algo.pop_size * pop_multiplier)
                        algo.mu = algo.pop_size // 2
                        weights = np.log(algo.mu + 0.5) - np.log(np.arange(1, algo.mu + 1))
                        algo.weights = weights / np.sum(weights)
                        algo.mueff = 1.0 / np.sum(algo.weights ** 2)
                    
                    # Exécuter optimisation
                    while algo.evals < max_evals:
                        solutions = algo.ask()
                        fitness = np.array([f(x) for x in solutions])
                        algo.tell(solutions, fitness)
                        algo.evals += len(solutions)
                    
                    function_results.append(algo.best_fitness)
                    function_curves.append(np.array(algo.history))
                    
                    # Mise à jour progress
                    progress = current_run / total_runs
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    eta = (elapsed / current_run) * (total_runs - current_run)
                    status_text.text(f"F{fid} - Run {run+1}/{num_runs} | Temps écoulé: {elapsed:.0f}s | ETA: {eta:.0f}s")
                
                # Stocker résultats pour cette fonction
                all_results[fid] = {
                    'mean': np.mean(function_results),
                    'std': np.std(function_results),
                    'min': np.min(function_results),
                    'max': np.max(function_results),
                    'all': function_results
                }
                all_curves[fid] = function_curves
            
            st.session_state.benchmark_results = all_results
            st.session_state.convergence_curves = all_curves
            st.session_state.benchmark_running = False
            
            progress_bar.progress(1.0)
            status_text.empty()
            
            st.success(f"✅ Benchmark terminé en {time.time() - start_time:.1f} secondes!")
            st.balloons()

# ============================================================================
# TAB 2: TABLEAU DES RÉSULTATS
# ============================================================================
with tabs[1]:
    st.header("📋 Tableau des Résultats")
    
    if 'benchmark_results' in st.session_state and st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        # Créer le tableau selon le format du projet
        data = {
            'Fonction': [],
            'Catégorie': [],
            'Moyenne': [],
            'Écart-type': [],
            'Min': [],
            'Max': []
        }
        
        for fid in sorted(results.keys()):
            data['Fonction'].append(f"F{fid}")
            
            if fid <= 3:
                data['Catégorie'].append("Unimodale")
            elif fid <= 10:
                data['Catégorie'].append("Multimodale")
            elif fid <= 20:
                data['Catégorie'].append("Hybride")
            else:
                data['Catégorie'].append("Composée")
            
            data['Moyenne'].append(f"{results[fid]['mean']:.2e}")
            data['Écart-type'].append(f"{results[fid]['std']:.2e}")
            data['Min'].append(f"{results[fid]['min']:.2e}")
            data['Max'].append(f"{results[fid]['max']:.2e}")
        
        df = pd.DataFrame(data)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Format tableau projet (comme demandé dans Description mini projet.md)
        st.subheader("📝 Format Tableau Projet")
        st.markdown("Tableau au format demandé dans le projet :")
        
        # Créer tableau transposé par groupes de 5
        functions = sorted(results.keys())
        
        for i in range(0, len(functions), 5):
            group = functions[i:i+5]
            
            headers = [f"F{fid}" for fid in group]
            means = [f"{results[fid]['mean']:.2e}" for fid in group]
            stds = [f"{results[fid]['std']:.2e}" for fid in group]
            
            group_df = pd.DataFrame({
                '': ['Moyenne', 'Écart-type'],
                **{h: [m, s] for h, m, s in zip(headers, means, stds)}
            })
            
            st.dataframe(group_df, use_container_width=True, hide_index=True)
            st.write("")
    else:
        st.info("Aucun résultat disponible. Lancez le benchmark dans l'onglet 'Exécuter Benchmark'.")

# ============================================================================
# TAB 3: COURBES DE CONVERGENCE
# ============================================================================
with tabs[2]:
    st.header("📈 Courbes de Convergence")
    
    if 'convergence_curves' in st.session_state and st.session_state.convergence_curves:
        curves = st.session_state.convergence_curves
        
        st.markdown("""
        Les courbes de convergence sont demandées pour les fonctions **F2, F4, F12, F25** selon le projet.
        """)
        
        # Sélection des fonctions à afficher
        available_functions = sorted(curves.keys())
        default_display = [f for f in [2, 4, 12, 25] if f in available_functions]
        
        selected_for_display = st.multiselect(
            "Fonctions à afficher",
            available_functions,
            default=default_display if default_display else available_functions[:4],
            format_func=lambda x: f"F{x}"
        )
        
        if selected_for_display:
            # Créer subplot grid
            n_funcs = len(selected_for_display)
            cols = 2
            rows = (n_funcs + 1) // 2
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"F{fid}" for fid in selected_for_display],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            colors = ['#4F46E5', '#22C55E', '#EF4444', '#F59E0B', '#8B5CF6', '#EC4899']
            
            for idx, fid in enumerate(selected_for_display):
                row = idx // 2 + 1
                col = idx % 2 + 1
                
                func_curves = curves[fid]
                
                # Calculer moyenne et écart-type des courbes
                # Normaliser les longueurs
                min_len = min(len(c) for c in func_curves)
                normalized_curves = np.array([c[:min_len] for c in func_curves])
                
                mean_curve = np.mean(normalized_curves, axis=0)
                std_curve = np.std(normalized_curves, axis=0)
                
                iterations = np.arange(len(mean_curve))
                
                # Ajouter bande d'écart-type
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([iterations, iterations[::-1]]),
                        y=np.concatenate([mean_curve + std_curve, (mean_curve - std_curve)[::-1]]),
                        fill='toself',
                        fillcolor=f'rgba(79, 70, 229, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'F{fid} ±σ',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Ajouter courbe moyenne
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=mean_curve,
                        mode='lines',
                        name=f'F{fid}',
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=400 * rows,
                template="plotly_white",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Échelle log pour l'axe y
            for i in range(1, n_funcs + 1):
                fig.update_yaxes(type="log", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Courbe comparative (toutes les fonctions sur un même graphique)
            st.subheader("📊 Comparaison Globale")
            
            fig_compare = go.Figure()
            
            for idx, fid in enumerate(selected_for_display):
                func_curves = curves[fid]
                min_len = min(len(c) for c in func_curves)
                normalized_curves = np.array([c[:min_len] for c in func_curves])
                mean_curve = np.mean(normalized_curves, axis=0)
                
                fig_compare.add_trace(
                    go.Scatter(
                        x=np.arange(len(mean_curve)),
                        y=mean_curve,
                        mode='lines',
                        name=f'F{fid}',
                        line=dict(color=colors[idx % len(colors)], width=2)
                    )
                )
            
            fig_compare.update_layout(
                title="Comparaison des courbes de convergence (moyenne sur 30 runs)",
                xaxis_title="Itérations",
                yaxis_title="Fitness (échelle log)",
                yaxis_type="log",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("Aucune courbe disponible. Lancez le benchmark dans l'onglet 'Exécuter Benchmark'.")

# ============================================================================
# TAB 4: EXPORTER RÉSULTATS
# ============================================================================
with tabs[3]:
    st.header("📥 Exporter les Résultats")
    
    if 'benchmark_results' in st.session_state and st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        st.markdown("""
        Exportez vos résultats pour le rapport du projet.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Format CSV")
            
            # Créer DataFrame pour export
            export_data = {
                'Fonction': [],
                'Catégorie': [],
                'Moyenne': [],
                'Écart-type': [],
                'Min': [],
                'Max': []
            }
            
            for fid in sorted(results.keys()):
                export_data['Fonction'].append(f"F{fid}")
                
                if fid <= 3:
                    export_data['Catégorie'].append("Unimodale")
                elif fid <= 10:
                    export_data['Catégorie'].append("Multimodale")
                elif fid <= 20:
                    export_data['Catégorie'].append("Hybride")
                else:
                    export_data['Catégorie'].append("Composée")
                
                export_data['Moyenne'].append(results[fid]['mean'])
                export_data['Écart-type'].append(results[fid]['std'])
                export_data['Min'].append(results[fid]['min'])
                export_data['Max'].append(results[fid]['max'])
            
            df_export = pd.DataFrame(export_data)
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name="cmaes_cec2017_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.subheader("📋 Format LaTeX")
            
            # Générer tableau LaTeX
            latex_table = "\\begin{tabular}{|l|" + "c|" * min(5, len(results)) + "}\n\\hline\n"
            
            functions = sorted(results.keys())
            for i in range(0, len(functions), 5):
                group = functions[i:i+5]
                
                # En-tête
                headers = " & ".join([f"F{fid}" for fid in group])
                latex_table += f" & {headers} \\\\\n\\hline\n"
                
                # Moyenne
                means = " & ".join([f"{results[fid]['mean']:.2e}" for fid in group])
                latex_table += f"Moyenne & {means} \\\\\n"
                
                # Écart-type
                stds = " & ".join([f"{results[fid]['std']:.2e}" for fid in group])
                latex_table += f"Écart-type & {stds} \\\\\n\\hline\n"
            
            latex_table += "\\end{tabular}"
            
            st.code(latex_table, language="latex")
            
            st.download_button(
                label="📥 Télécharger LaTeX",
                data=latex_table,
                file_name="cmaes_cec2017_results.tex",
                mime="text/plain",
                use_container_width=True
            )
        
        st.divider()
        
        st.subheader("📊 Résumé Statistique")
        
        # Statistiques globales par catégorie
        categories = {
            'Unimodale': [fid for fid in results.keys() if fid <= 3],
            'Multimodale': [fid for fid in results.keys() if 4 <= fid <= 10],
            'Hybride': [fid for fid in results.keys() if 11 <= fid <= 20],
            'Composée': [fid for fid in results.keys() if fid >= 21]
        }
        
        summary_data = []
        for cat_name, fids in categories.items():
            if fids:
                means = [results[fid]['mean'] for fid in fids]
                summary_data.append({
                    'Catégorie': cat_name,
                    'Nb Fonctions': len(fids),
                    'Moyenne globale': f"{np.mean(means):.2e}",
                    'Meilleure': f"{np.min(means):.2e}",
                    'Pire': f"{np.max(means):.2e}"
                })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun résultat à exporter. Lancez le benchmark d'abord.")

st.divider()

# Note sur le benchmark officiel
if CEC2017_OFFICIAL:
    st.success("✅ **Benchmark officiel CEC2017 actif** - Les résultats sont conformes aux standards académiques.")
else:
    st.warning("⚠️ Utilisation d'une implémentation simplifiée. Pour des résultats officiels, installez le package cec2017.")
