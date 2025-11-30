# Description mini projet « Résolution du Benchmark CEC2017 »

**Section I2 (Décembre 2025)**

## Sujet

Comme expliqué en classe, on s'intéresse dans ce mini-projet aux problèmes continus.

Nous considérons le benchmark CEC 2017 dont le code python est disponible sur :

<https://github.com/tilleyd/cec2017-py>

Ce benchmark contient 30 fonctions se divisant en quatre groupes :

- Les fonctions f1 à f3 sont uni-modales ;
- Les fonctions f4 à f10 sont multimodales ;
- Les fonctions f11 à f20 sont hybrides ;
- Les fonctions f21 à f30 sont composées.

Plus de détails sur ces fonctions sont dans l'Annexe A.  
**Les tests sont à réaliser en dimensions D = 30. Chaque algorithme est exécuté 30 fois pour chaque fonction avec un critère d'arrêt un nombre maximum d'évaluations de la fonction objectif égal à 10000\*D = 30000 évaluations.**

## Objectifs et consignes

1. Evaluer les acquis des étudiants et leurs capacités à comparer plusieurs méthodes vues en classes et à les affiner avec des jeux de paramètres, des hybridations, ou des recherches locales.

2. Evaluer l'innovation des étudiants et la distinction de leur travail par rapport aux autres y compris l'utilisation de méthodes non vues en classe.

3. Quoique le travail est en binôme, les notes peuvent différer drastiquement puisque je poserai des questions individuelles pour tester l'**implication** de chaque étudiant dans le travail.

4. Les étudiants présenteront leurs méthodes et codes et remettront en papier la description des méthodes testées, le **tableau des résultats** et les graphiques (courbes de convergences en couleur). En gros 4 pages suffisent.

5. Chaque binôme aura 20 minutes (présentation rapide + questions)

## Travail demandé

- Tester les méthodes sur le benchmark en prenant le critère d'arrêt est le **nombre maximum d'évaluations** de la fonction objectif = **30000** (ajouter le compteur nécessaire).

- Attention à ne pas modifier les fonctions du benchmark sinon vos résultats **pourront être erronées**.

- **Votre algorithme** doit être testé **30 fois** sur chaque fonction et vous rapporter le tableau des résultats contenant la **moyenne** et **l'écart type**.

|            | F1       | F2       | F3       | F4       | F5       |
|------------|----------|----------|----------|----------|----------|
| Moyenne    |          |          |          |          |          |
| Ecart-type |          |          |          |          |          |
| ...        |          |          |          |          |          |
|            | F26      | F27      | F28      | F29      | F30      |
| Moyenne    |          |          |          |          |          |
| Ecart-type |          |          |          |          |          |

- Les figures de convergence pour les fonctions f2, f4, f12 et f25. Ici vous allez comparer par rapport à d'autres méthodes vues en classe.

![Courbes de convergence](figures.png)