# Système d’Aide à la Décision (SAD) - Machine Learning
## Diagnostic de Tumeurs Cérébrales par IRM
Machine Learning – ALIF83 (2025–2026)

Yann Renvoisé, Aïssa Mehenni, LSI2

---

## Objectif du projet

Ce projet consiste à développer un **Système d’Aide à la Décision (SAD)** pour assister un service de radiologie dans le diagnostic de tumeurs cérébrales à partir d’images IRM.

Contrairement à un simple classificateur, le système :

- Produit un diagnostic
- Estime un niveau de confiance
- Applique des règles médicales basées sur des seuils
- Priorise les cas urgents
- Minimise les faux négatifs (risque vital)
- Génère un rapport clinique automatisé

Il s’agit d’un outil d’aide à la décision et non d’un remplacement du radiologue.

---

## Dataset

Dataset utilisé :  
Brain Tumor MRI Dataset (Kaggle)

Lien :  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Classes :

- glioma → tumeur maligne (URGENT)
- meningioma → tumeur bénigne (surveillance)
- pituitary → traitement spécialisé
- notumor → rassurer le patient

---

## Structure du projet

```text
Projet_SAD/
│
├── data/ (non versionné – à télécharger séparément)
│ ├── Training/
│ └── Testing/
│
├── notebooks/
│ └── SAD_brain_tumor.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ ├── calibration.py
│ ├── decision_engine.py
│ ├── uncertainty.py
│ ├── reporting.py
│ └── evaluation.py
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Cloner le projet

```shell
git clone <https://github.com/YannRenvoise/Projet_SAD>
cd Projet_SAD
```

### Installer les dépendances

Créer un environnement virtuel recommandé :
```shell
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

Puis :
```shell
pip install -r requirements.txt
```

---

## Installation des données

1. Télécharger le dataset depuis Kaggle
2. Dézipper le dossier
3. Placer les dossiers `Training` et `Testing` dans :

```shell
Projet_SAD/data/
```

Le dossier `data/` n’est pas versionné dans Git.

---

## Pipeline du système

IRM → Prétraitement → Modèle (RegLog / MLP / CNN calibré)  
→ Probabilités  
→ Moteur de décision clinique  
→ Rapport automatisé

---

## Modèles implémentés

- Régression Logistique multinomiale + Calibration (Platt / Isotonic)
- MLP avec sortie probabiliste
- CNN + Temperature Scaling
- Estimation d’incertitude via Monte Carlo Dropout

---

## Moteur de Décision

Seuils de confiance :

- ≥ 0.85 → Diagnostic automatique validé
- ≥ 0.65 → Révision recommandée
- ≥ 0.50 → Cas incertain
- < 0.50 → Double lecture obligatoire

Gestion asymétrique des faux négatifs :
Si prédiction = "notumor" avec probabilité < 0.95 → Vérification obligatoire.

---

## Métriques métier

- Taux de couverture automatique
- Accuracy par tranche de confiance
- Analyse coût-bénéfice :

Coût = (FN × 1000) + (FP × 100) + (Révision × 50)

---

## Livrables

- Notebook complet
- Implémentation des fonctions SAD
- 20 rapports simulés
- Analyse critique et éthique

---

## Éthique

Ce système est un outil d’assistance.  
Il ne remplace pas un professionnel de santé.

---

## Auteurs

Yann Renvoisé, Aïssa Mehenni
Projet académique – 2025–2026