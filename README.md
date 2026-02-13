# Systeme d'Aide a la Decision (SAD) - Machine Learning
## Diagnostic de tumeurs cerebrales sur IRM
Module ALIF83 - 2025-2026

Ce depot implemente un SAD academique conforme au cahier des charges du projet:
- modeles probabilistes (RegLog, MLP, CNN)
- calibration des probabilites (isotonic/sigmoid, temperature scaling)
- moteur de decision clinique a seuils multiples
- gestion du risque faux negatif "notumor"
- metriques metier et analyse cout-benefice
- generation automatisee de rapports textuels

## 1. Installation
```bash
python -m venv venv
source venv/bin/activate            # Linux/Mac
# venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

## 2. Donnees
Dataset utilise:
[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Placez les donnees ainsi:
```text
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

`data/` est ignore par Git.

## 3. Structure du projet
```text
Projet_SAD/
├── notebooks/
│   └── SAD_brain_tumor.ipynb
├── reports/
│   └── sample_reports.txt
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── calibration.py
│   ├── decision_engine.py
│   ├── uncertainty.py
│   ├── evaluation.py
│   └── reporting.py
├── requirements.txt
└── README.md
```

## 4. Correspondance avec le cahier des charges
Fonctions attendues et emplacement:
- `predire_avec_confiance(...)` -> `src/decision_engine.py`
- `generer_recommandation(...)` -> `src/decision_engine.py`
- `calculer_incertitude_mc_dropout(...)` -> `src/uncertainty.py`
- `creer_rapport_decision(...)` -> `src/reporting.py`

Elements demandes:
- RegLog calibree: `src/models.py`, `src/calibration.py`
- MLP probabiliste: `src/models.py`
- CNN + temperature scaling: `src/models.py`, `src/calibration.py`
- Seuils de decision clinique: `src/decision_engine.py`
- Metriques metier SAD: `src/evaluation.py`
- Rapport automatise patient: `src/reporting.py`

## 5. Execution recommandee
Le livrable principal est le notebook:
`notebooks/SAD_brain_tumor.ipynb`

Il couvre:
1. contexte SAD (classification vs decision)
2. entrainement RegLog + calibration
3. entrainement MLP + analyse d'incertitude
4. entrainement CNN + calibration temperature
5. moteur de decision clinique
6. metriques metier (couverture, acc haute confiance, cout)
7. generation de 20 rapports
8. analyse critique et ethique

## 6. Hypotheses cliniques utilisees
- classes: `glioma`, `meningioma`, `pituitary`, `notumor`
- seuils:
  - `>= 0.85`: diagnostic automatique valide
  - `>= 0.65`: revision recommandee
  - `>= 0.50`: cas incertain
  - `< 0.50`: double lecture obligatoire
- regle de securite:
  - si classe predite `notumor` et confiance `< 0.95`,
    verification humaine obligatoire

## 7. Attention academique
Ce projet est un exercice de cours. Ce SAD ne remplace pas un radiologue.
Toute decision clinique reelle exige validation medicale humaine.
