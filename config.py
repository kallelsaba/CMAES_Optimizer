# Configuration globale - LIGHT THEME

DIMENSION = 30
MAX_EVALUATIONS = 30000
BOUNDS = [-100, 100]

# Couleurs thème clair moderne
PRIMARY_COLOR = "#4F46E5"      # Indigo
SECONDARY_COLOR = "#7C3AED"    # Violet
ACCENT_COLOR = "#EC4899"       # Rose
BACKGROUND = "#FFFFFF"         # Blanc
SURFACE = "#F8FAFC"            # Gris très clair
TEXT_PRIMARY = "#1E293B"       # Texte foncé
TEXT_SECONDARY = "#64748B"     # Texte gris

# CMA-ES paramètres par défaut
DEFAULT_CMA_PARAMS = {
    "population": 30,
    "sigma": 0.3,
    "c_c": 0.4,
    "c_s": 0.3,
    "c1": 0.02,
    "c_mu": 0.3,
}

# CEC2017 fonction labels
CEC_FUNCTIONS = {
    "F1-F3": "Unimodale",
    "F4-F10": "Multimodale",
    "F11-F20": "Hybride",
    "F21-F30": "Composée"
}