# Système de Surveillance des Signes Vitaux et de Détection d'Anomalies Basé sur l'IoT

### **Comment exécuter le projet**

```markdown
**Naviguez vers le répertoire de votre projet**
```bash
cd path/to/your/project
```

**Activez l'environnement virtuel**
```bash
myenv\Scripts\activate
```

**Installez les dépendances**
```bash
pip install -r requirements.txt
```

**Si vous avez une carte graphique NVIDIA**
```bash
pip install tensorflow-gpu
```

**Si vous avez terminé et souhaitez désactiver l'environnement**
```bash
deactivate
```

**Collectez les données**
```bash
python DataGenerator.py
```

**Ajustez les horodatages si vous exécutez le programme de génération de données en parallèle**
```bash
python AdjustTimeStamp.py
```

**Entraînez le modèle**
```bash
python modelTraining.py
```

**Testez le modèle**
```bash
python RunTheModel.py
```

## Vue d'ensemble

Ce projet a pour but de mettre en place un système IoT (Internet des Objets) sophistiqué permettant de surveiller les signes vitaux, tels que la pression artérielle et la fréquence cardiaque. Il utilise des techniques d'intelligence artificielle pour gérer les ressources et détecter les anomalies. Le système comprend un générateur de données réaliste, un modèle de prédiction haute précision basé sur l'architecture BiLSTM, et une interface utilisateur intuitive pour la visualisation et le contrôle en temps réel.

**Auteur du Projet:** Mouhamed Wassim Mbarek

**Classe :** LCE IoT 3A Groupe 2

**Encadré par :** Mme. Yesmine Gara

## Table des Matières

- [Résumé Exécutif](#résumé-exécutif)
- [Réalisations](#réalisations)
- [Points Forts](#points-forts)
- [Limites et Améliorations Futures](#limites-et-améliorations-futures)
- [Perspectives Futures](#perspectives-futures)
- [Introduction](#introduction)
- [Objectifs du Projet](#objectifs-du-projet)
- [Collecte de Données](#collecte-de-données)
  - [Génération de Données](#génération-de-données)
  - [Paramètres Physiologiques](#paramètres-physiologiques)
  - [Scénarios d'Anomalies](#scénarios-danomalies)
  - [Contraintes Physiologiques](#contraintes-physiologiques)
  - [Exemple de Données Générées](#exemple-de-données-générées)
- [Interface Graphique](#interface-graphique)
- [Conclusion sur la génération de données](#conclusion-sur-la-génération-de-données)
- [Fonctionnement du Code](#fonctionnement-du-code)
- [Entraînement, Architecture et Configuration du Modèle](#entraînement-architecture-et-configuration-du-modèle)
  - [Structure du Réseau](#structure-du-réseau)
  - [Paramètres d'Entraînement](#paramètres-dentraînement)
  - [Convergence du Modèle](#convergence-du-modèle)
  - [Métriques de Performance Finales](#métriques-de-performance-finales)
  - [Courbe de Précision](#courbe-de-précision)
  - [Courbe de Perte](#courbe-de-perte)
  - [Évaluation des Performances : Analyse ROC](#évaluation-des-performances--analyse-roc)
  - [Analyse de la Matrice de Confusion : Performance par Classe](#analyse-de-la-matrice-de-confusion--performance-par-classe)
- [Points Forts et Limitations](#points-forts-et-limitations)
  - [Points Forts](#points-forts-1)
  - [Limitations](#limitations)
- [Conclusions](#conclusions)
- [Métriques Détaillées](#métriques-détaillées)
  - [Métriques Globales](#métriques-globales)
  - [Temps et Ressources](#temps-et-ressources)
- [Tests et Évaluation du Modèle en Temps Réel](#tests-et-évaluation-du-modèle-en-temps-réel)
  - [Justification de la Limitation à 10 Points de Données](#justification-de-la-limitation-à-10-points-de-données)
  - [Résultats des Tests avec Fenêtre Limitée](#résultats-des-tests-avec-fenêtre-limitée)
  - [Analyse des Performances](#analyse-des-performances)
- [Conclusion Finale](#conclusion-finale)

## Résumé Exécutif

Ce projet a permis de concevoir et de mettre en œuvre un système IoT avancé pour la surveillance des paramètres vitaux. Il intègre des techniques d'intelligence artificielle pour la gestion des ressources et la détection d'anomalies. Les objectifs initiaux, à savoir la prédiction des besoins en ressources, la détection d'anomalies et la visualisation des données en temps réel, ont été atteints avec succès.

## Réalisations

-   **Génération de Données Réalistes :** Le programme `HealthDataGenerator` simule fidèlement les signes vitaux, reproduisant le comportement du corps humain dans des conditions normales et pathologiques. Il intègre des paramètres physiologiques configurables et des scénarios d'anomalies prédéfinis.
-   **Modèle de Prédiction Performant :** Un modèle d'apprentissage automatique basé sur l'architecture BiLSTM a été entraîné avec succès. Il atteint une précision globale supérieure à 98% et peut distinguer différentes conditions cardiaques (normale, arrêt cardiaque, crise hypertensive, choc septique).
-   **Tests en Conditions Réelles :** Une interface graphique a permis de tester le modèle en temps réel avec des données simulées. Même avec une fenêtre de données limitée à 10 points, le modèle a maintenu une précision de 93%, ce qui souligne sa robustesse et sa réactivité.
-   **Interface Utilisateur Intuitive :** L'interface graphique offre une visualisation claire et en temps réel des données de signes vitaux et permet d'ajuster les paramètres de simulation. Elle affiche l'état actuel (normal ou anomalie détectée) et la précision du modèle.

## Points Forts

-   **Approche Intégrée :** Le projet combine efficacement la génération de données physiologiques, la modélisation par apprentissage automatique et la visualisation en temps réel, offrant une solution complète pour la surveillance des paramètres vitaux.
-   **Haute Précision :** Le modèle d'apprentissage automatique a démontré une précision exceptionnelle, tant lors de l'entraînement que lors des tests en conditions réelles.
-   **Robustesse et Réactivité :** Les tests avec une fenêtre de données limitée ont prouvé la capacité du modèle à s'adapter rapidement aux changements et à maintenir une bonne précision, même avec un historique de données minimal.
-   **Simulateur Réaliste :** `HealthDataGenerator` est un outil précieux pour la génération de données de santé. Il est utile pour l'entraînement de modèles d'IA et la simulation de divers scénarios médicaux.

## Limites et Améliorations Futures

-   **Dépendance aux Données Simulées :** Le système n'a pas encore été testé avec des données réelles provenant de capteurs médicaux.
-   **Complexité du Modèle :** Le modèle BiLSTM, bien que précis, est relativement complexe et pourrait nécessiter des ressources de calcul importantes pour un déploiement à grande échelle.
-   **Amélioration de l'Interface Utilisateur :** L'interface graphique pourrait être enrichie avec des fonctionnalités supplémentaires, telles que des alertes personnalisables ou des rapports d'analyse plus détaillés.

## Perspectives Futures

-   **Intégration de Données Réelles :** Intégrer des données réelles provenant de capteurs médicaux pour valider et améliorer le modèle dans des conditions cliniques réelles.
-   **Déploiement sur un Système Embarqué :** Déployer le système sur un dispositif IoT embarqué, en optimisant le modèle pour une consommation énergétique réduite.
-   **Détection d'Anomalies Plus Complexes :** Entraîner le modèle à détecter des anomalies plus subtiles ou des combinaisons d'anomalies, augmentant ainsi sa valeur diagnostique.
-   **Intégration d'un Système d'Alerte :** Intégrer un système d'alerte en temps réel pour avertir les professionnels de santé en cas d'anomalies critiques.
-   **Personnalisation du Modèle :** Personnaliser le modèle en fonction des caractéristiques individuelles des patients pour améliorer encore la précision des prédictions.

## Introduction

L'objectif de ce projet est de développer un système IoT innovant, capable de surveiller des paramètres vitaux tels que la pression sanguine et la fréquence cardiaque. Le tout en utilisant des techniques avancées d'intelligence artificielle pour optimiser la gestion des ressources.

## Objectifs du Projet

-   Prédire les besoins en ressources.
-   Détecter les anomalies.
-   Visualiser les données en temps réel.

## Collecte de Données

### Génération de Données

Le programme Python `HealthDataGenerator` a été développé pour simuler de manière réaliste le comportement du corps humain dans diverses conditions. Il génère des données synthétiques de signes vitaux en se basant sur des paramètres physiologiques configurables et des scénarios d'anomalies prédéfinis.

### Paramètres Physiologiques

Le programme utilise les paramètres de base suivants pour simuler les signes vitaux dans des conditions normales :

-   **Pression artérielle :**
    -   Systolique : moyenne de 120 mmHg, écart-type de 5 mmHg, minimum de 70 mmHg, maximum de 180 mmHg.
    -   Diastolique : moyenne de 80 mmHg, écart-type de 3 mmHg, minimum de 40 mmHg, maximum de 120 mmHg.
-   **Fréquence cardiaque :** moyenne de 75 bpm, écart-type de 3 bpm, minimum de 40 bpm, maximum de 150 bpm. Une variation naturelle de la fréquence cardiaque liée à la respiration (arythmie sinusale respiratoire) est également simulée avec un facteur de 0.1.
-   **SpO2 (Saturation pulsée en oxygène) :** moyenne de 98%, écart-type de 0.5%, minimum de 80%, maximum de 100%.
-   **Fréquence respiratoire :** moyenne de 16 cycles/min, écart-type de 1 cycle/min, minimum de 8 cycles/min, maximum de 30 cycles/min.

Ces paramètres sont basés sur des standards médicaux pour garantir le réalisme des données générées.

### Scénarios d'Anomalies

Le programme est capable de simuler différentes anomalies avec une probabilité configurable (par défaut 3%). Les anomalies implémentées sont :

-   **Arrêt cardiaque :**
    -   Diminution rapide de la pression artérielle (systolique : -40 mmHg, diastolique : -30 mmHg) avec un écart-type de 15 mmHg.
    -   Chute de la fréquence cardiaque (-30 bpm) avec un écart-type de 20 bpm.
    -   Baisse de la SpO2 (-15%) avec un écart-type de 5%.
    -   Augmentation de la fréquence respiratoire (+8 cycles/min) avec un écart-type de 3 cycles/min.
    -   Durée de l'anomalie : 15 secondes.
    -   Début : rapide.
-   **Crise hypertensive :**
    -   Augmentation graduelle de la pression artérielle (systolique : +60 mmHg, diastolique : +40 mmHg) avec un écart-type de 10 mmHg.
    -   Hausse de la fréquence cardiaque (+30 bpm) avec un écart-type de 15 bpm.
    -   Légère baisse de la SpO2 (-5%) avec un écart-type de 2%.
    -   Augmentation de la fréquence respiratoire (+6 cycles/min) avec un écart-type de 2 cycles/min.
    -   Durée de l'anomalie : 20 secondes.
    -   Début : progressif.
-   **Choc septique :**
    -   Diminution progressive de la pression artérielle (systolique : -30 mmHg, diastolique : -20 mmHg) avec un écart-type de 10 mmHg.
    -   Augmentation importante de la fréquence cardiaque (+40 bpm) avec un écart-type de 10 bpm.
    -   Baisse de la SpO2 (-10%) avec un écart-type de 3%.
    -   Augmentation de la fréquence respiratoire (+10 cycles/min) avec un écart-type de 3 cycles/min.
    -   Durée de l'anomalie : 25 secondes.
    -   Début : progressif.

### Contraintes Physiologiques

Afin d'assurer un réalisme accru, le programme applique des contraintes physiologiques aux données générées. Par exemple :

-   Une fréquence cardiaque élevée entraîne une augmentation de la pression artérielle.
-   Une SpO2 inférieure à 90% provoque une augmentation de la fréquence cardiaque.

### Exemple de Données Générées

Les données sont générées à un intervalle de 0.5 seconde et enregistrées dans un fichier CSV. Voici un exemple de données générées par le programme :

| Horodatage                  | Pression Systolique | Pression Diastolique | Fréquence Cardiaque | SpO2  | Fréquence Respiratoire | Type d'Anomalie |
| :-------------------------- | :------------------ | :------------------- | :------------------ | :---- | :--------------------- | :-------------- |
| 2024-12-27 12:01:35.702104  | 122.86              | 76.46                | 77.45               | 97.18 | 15.96                  | normal          |
| 2024-12-27 12:01:36.213229  | 119.16              | 83.21                | 75.34               | 97.52 | 16.75                  | normal          |
| 2024-12-27 12:01:36.718694  | 116.57              | 83.63                | 75.36               | 98.26 | 14.08                  | normal          |
| 2024-12-27 12:01:37.226898  | 120.30              | 82.62                | 78.83               | 97.22 | 15.40                  | normal          |
| 2024-12-27 12:01:37.732541  | 116.82              | 80.62                | 72.77               | 97.91 | 15.87                  | normal          |
| 2024-12-27 12:01:40.833008  | 128.96              | 78.60                | 74.30               | 100.00| 15.08                  | choc_septique   |
| 2024-12-27 12:01:40.936811  | 114.03              | 71.73                | 72.27               | 98.33 | 18.87                  | choc_septique   |
| 2024-12-27 12:01:41.044036  | 122.64              | 92.97                | 62.50               | 94.72 | 16.76                  | choc_septique   |
| 2024-12-27 12:01:41.147348  | 136.96              | 71.08                | 83.57               | 92.89 | 17.06                  | choc_septique   |
| 2024-12-27 12:01:41.250238  | 109.95              | 87.01                | 60.94               | 94.85 | 9.98                   | choc_septique   |

## Interface Graphique

Le programme `HealthDataGenerator` est accompagné d'une interface graphique qui permet de visualiser en temps réel les données générées et de configurer les paramètres de simulation. L'interface affiche les graphiques de la pression artérielle (systolique et diastolique), de la fréquence cardiaque, de la SpO2 et de la fréquence respiratoire. Elle permet également de modifier l'intervalle de génération des données, la taille de la fenêtre d'observation et la probabilité d'occurrence des anomalies.

## Conclusion sur la génération de données

Le programme `HealthDataGenerator` permet de générer des données de signes vitaux réalistes en simulant le comportement du corps humain dans des conditions normales et pathologiques. L'utilisation de paramètres physiologiques configurables, de scénarios d'anomalies prédéfinis et de contraintes physiologiques assure la cohérence et la pertinence des données générées. Ces données peuvent être utilisées pour l'entraînement et la validation de modèles d'apprentissage automatique pour la détection d'anomalies dans les données de santé.

## Fonctionnement du Code

Pour faire simple, le code simule la génération de données de signes vitaux, comme la pression artérielle, la fréquence cardiaque, la SpO2 et la fréquence respiratoire. Il inclut des variations normales et des anomalies telles que l'arrêt cardiaque, la crise hypertensive et le choc septique.

Il repose sur deux classes principales :

-   **`HealthDataGenerator` :** C'est le chef d'orchestre de la génération de données. Il configure les paramètres (fréquence de génération, probabilité d'anomalie, etc.), définit les valeurs normales et les caractéristiques des anomalies pour les signes vitaux. Il applique même des contraintes physiologiques pour plus de réalisme (par exemple, une SpO2 basse augmente la fréquence cardiaque). Enfin, il enregistre les données générées dans un fichier CSV, un peu comme un journal de bord.
-   **`HealthMonitorDashboard` :** C'est l'interface utilisateur graphique (GUI). Elle permet de visualiser les données en temps réel sous forme de jolis graphiques. On peut modifier les paramètres de simulation grâce à des curseurs et des boutons. Elle affiche l'état actuel (normal ou type d'anomalie) et les valeurs des signes vitaux, comme un tableau de bord médical.

Le flux de travail, c'est un peu comme une recette de cuisine :

1. **Initialisation :** On prépare les ingrédients. Configuration des paramètres et démarrage de la génération de données.
2. **Boucle de génération :** On touille la marmite !
    -   Génération de valeurs normales ou simulation d'anomalies, selon les paramètres.
    -   Application de contraintes physiologiques, pour que ça reste cohérent.
    -   Enregistrement des données dans le fichier CSV, pour garder une trace.
3. **Boucle d'interface utilisateur :** On surveille la cuisson !
    -   Mise à jour des graphiques et des informations affichées, en temps réel.
    -   Prise en compte des modifications de paramètres par l'utilisateur, si on change la recette en cours de route.
4. **Terminaison :** On coupe le feu ! Arrêt de la génération de données et fermeture de l'interface utilisateur, quand on a fini.

En résumé, le code crée un simulateur de signes vitaux configurable avec une interface graphique pour la visualisation et le contrôle. C'est un outil pratique pour générer des données, par exemple pour entraîner des modèles d'apprentissage automatique qui pourront ensuite détecter des anomalies.

## Entraînement, Architecture et Configuration du Modèle

### Structure du Réseau

Le modèle utilise une architecture BiLSTM (Bidirectional Long Short-Term Memory), un type de réseau neuronal particulièrement efficace pour analyser des séquences de données. Voici sa structure :

-   **Couche BiLSTM initiale :** 128 unités, avec retour de séquences (pour analyser dans les deux sens).
-   **Couche Dropout (0.2) :** Pour éviter le surapprentissage (un peu comme si on oubliait volontairement quelques informations pour mieux généraliser).
-   **Seconde couche BiLSTM :** 64 unités (toujours bidirectionnelle).
-   **Couche Dropout (0.2) :** Encore un peu d'oubli contrôlé.
-   **Couche Dense :** 64 unités avec activation ReLU (une fonction mathématique qui introduit de la non-linéarité).
-   **Couche Dropout (0.2) :** On continue de simplifier.
-   **Couche de sortie :** Dense avec activation softmax (pour obtenir des probabilités pour chaque classe : normal, arrêt cardiaque, etc.).

### Paramètres d'Entraînement

-   **Optimiseur :** Adam, avec un taux d'apprentissage de 0.001 (c'est lui qui ajuste les paramètres du modèle pendant l'entraînement).
-   **Fonction de perte :** Entropie croisée catégorielle épars (Sparse Categorical Crossentropy), pour mesurer l'erreur entre les prédictions et les vraies valeurs.
-   **Taille de lot (batch size) :** 32 (le nombre d'exemples utilisés pour chaque mise à jour des paramètres).
-   **Époques maximales :** 30 (le nombre de fois que le modèle voit l'ensemble des données d'entraînement).
-   **Validation split :** 0.2 (20% des données sont utilisées pour la validation, pour vérifier que le modèle généralise bien).
-   **Early Stopping :** Patience de 5 époques (si la performance sur les données de validation ne s'améliore pas pendant 5 époques, on arrête l'entraînement pour éviter le surapprentissage).

### Convergence du Modèle

L'entraînement s'est terminé après 15 époques sur les 30 prévues, grâce au mécanisme d'Early Stopping. Cela signifie que le modèle a convergé de manière optimale, évitant ainsi le surapprentissage. On a arrêté avant qu'il ne commence à apprendre par cœur les données d'entraînement sans pouvoir généraliser.

### Métriques de Performance Finales

-   **Précision d'entraînement :** 98.8% (le modèle a bien appris sur les données d'entraînement).
-   **Précision de validation :** 98.2% (le modèle généralise bien sur des données qu'il n'a pas vues pendant l'entraînement).
-   **Perte d'entraînement finale :** ~0.035 (une mesure de l'erreur sur les données d'entraînement).
-   **Perte de validation finale :** ~0.06 (une mesure de l'erreur sur les données de validation).

### Courbe de Précision

-   **Phase initiale :** Augmentation rapide jusqu'à environ 97.5% dans les 2 premières époques.
-   **Phase intermédiaire :** Progression graduelle de 97.5% à 98.5%.
-   **Phase finale :** Stabilisation autour de 98.8% pour l'entraînement.
-   **Écart train/validation :** Environ 0.6%, ce qui indique un bon équilibre (le modèle n'est ni trop simple, ni trop complexe).

### Courbe de Perte

-   **Diminution initiale rapide :** De ~0.13 à ~0.07 dans la première époque.
-   **Réduction progressive :** Atteignant ~0.035 pour l'entraînement.
-   **Stabilisation :** Convergence après l'époque 10.
-   **Comportement de la validation :** Suit la tendance de l'entraînement avec un écart acceptable.

### Évaluation des Performances : Analyse ROC

Le modèle obtient des performances exceptionnelles avec une AUC (Area Under the Curve) de 1.00 pour toutes les classes (Arrêt cardiaque, Crise hypertensive, État normal, Choc septique). Cela signifie qu'il est capable de parfaitement distinguer les différentes classes.

### Analyse de la Matrice de Confusion : Performance par Classe

| État               | Prédictions Correctes | Faux Positifs | Faux Négatifs | Précision |
| :----------------- | :-------------------- | :------------ | :------------ | :-------- |
| Normal             | 10,426                | 25            | 121           | 99.76%    |
| Arrêt Cardiaque    | 5,073                 | 99            | 97            | 98.12%    |
| Crise Hypertensive | 7,088                 | 169           | 130           | 97.89%    |
| Choc Septique      | 8,345                 | 190           | 135           | 97.82%    |

## Points Forts et Limitations

### Points Forts

-   **Excellente généralisation :** L'écart minimal entre les performances d'entraînement et de validation montre que le modèle ne se contente pas d'apprendre par cœur les données d'entraînement.
-   **ROC parfait :** Indique une séparation optimale des classes. Le modèle est très bon pour distinguer les différentes conditions.
-   **Convergence rapide et stable :** L'entraînement est efficace et ne nécessite pas un nombre excessif d'époques.
-   **Taux de faux positifs très bas :** Pour toutes les classes, ce qui est crucial dans un contexte médical.

### Limitations

-   **Légère tendance à la surconfiance :** Le modèle est parfois un peu trop sûr de ses prédictions.
-   **Petit déséquilibre dans la performance entre les classes :** La performance est légèrement meilleure pour la classe "Normal" que pour les autres.
-   **Potentiel de surapprentissage :** L'arrêt précoce a été nécessaire pour éviter le surapprentissage, ce qui montre qu'il faut rester vigilant.

## Conclusions

Le modèle démontre une performance exceptionnelle avec une précision globale supérieure à 98% et une capacité remarquable à distinguer les différentes conditions cardiaques. L'arrêt précoce à l'époque 15 confirme une convergence efficace et évite le surapprentissage. C'est un excellent résultat !

## Métriques Détaillées

### Métriques Globales

-   **Accuracy :** 98.34% (la précision globale du modèle).
-   **Macro F1-Score :** 0.979 (une moyenne des F1-Scores pour chaque classe, utile quand les classes sont déséquilibrées).
-   **Micro F1-Score :** 0.983 (une autre moyenne des F1-Scores, moins sensible au déséquilibre des classes).
-   **Kappa Score :** 0.977 (une mesure de l'accord entre les prédictions et les vraies valeurs, qui prend en compte le hasard).

### Temps et Ressources

-   **Temps d'entraînement total :** 15 époques.
-   **Temps moyen par époque :** Environ 45 secondes.
-   **Utilisation maximale de la mémoire :** Environ 4.2 GB.
-   **Utilisation GPU :** Optimisée avec une croissance de mémoire contrôlée (pour utiliser efficacement les ressources du GPU).

## Tests et Évaluation du Modèle en Temps Réel

Afin de tester rigoureusement la réactivité et la précision du modèle dans des conditions de données limitées, la taille de la fenêtre d'observation a été réduite à seulement 10 points de données consécutifs. L'objectif est de simuler des situations où seules les informations les plus récentes sont disponibles. Ainsi, le modèle est forcé de prendre des décisions basées sur un historique extrêmement court.

### Justification de la Limitation à 10 Points de Données

-   **Évaluation de la Réactivité :** Un historique de 10 points permet d'évaluer la capacité du modèle à détecter rapidement des changements de tendance. C'est crucial pour des situations critiques nécessitant une intervention immédiate.
-   **Test de Robustesse :** Limiter les données d'entrée pousse le modèle dans ses retranchements. On teste ainsi sa robustesse face à un flux d'informations minimaliste.
-   **Simulation de Scénarios Réalistes :** Dans certains contextes médicaux, l'accès à un historique complet des données peut être impossible. Il est donc crucial que le modèle puisse fonctionner efficacement avec des informations limitées.

### Résultats des Tests avec Fenêtre Limitée

Malgré cette contrainte sévère, le modèle a démontré une performance remarquable, atteignant une précision moyenne de 93% dans la détection des anomalies. C'est très encourageant ! Cela suggère que le modèle est capable de généraliser efficacement à partir d'un nombre très limité d'observations.

### Analyse des Performances

-   **Adaptabilité :** Le modèle s'adapte rapidement aux changements de conditions, même avec seulement 10 points de données.
-   **Sensibilité aux Anomalies :** La limitation des données n'a pas significativement affecté la capacité du modèle à identifier les anomalies, ce qui démontre une bonne sensibilité.

## Conclusion Finale

Ce projet a pour objectif de développer un système IoT avancé pour surveiller les paramètres vitaux, tels que la pression sanguine et la fréquence cardiaque. Il intègre des techniques d'intelligence artificielle pour la gestion des ressources et la détection des anomalies.

Le projet s'articule autour de trois axes principaux :

-   **Prédiction des ressources :** Des modèles de machine learning sont utilisés pour anticiper la consommation énergétique des capteurs IoT. Ils sont évalués avec des métriques comme le RMSE (Root Mean Square Error).
-   **Détection des anomalies :** Des réseaux neuronaux (autoencodeurs, RNN) sont employés pour identifier les comportements anormaux dans les données des capteurs.
-   **Ajustement des ressources :** Un agent de reinforcement learning est utilisé pour optimiser dynamiquement la gestion des ressources en temps réel.

En conclusion, ce projet est une belle réussite. Il pose les bases d'un système de surveillance médicale performant et prometteur !