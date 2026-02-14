# Système d'Aide à la Décision (SAD) pour le Diagnostic de Tumeurs Cérébrales

Projet académique du module **Machine Learning (ALIF83)** - Année **2025-2026**

## Informations générales
- **Enseignant** : Mohamed HAMIDI
- **Étudiants** : Aïssa MEHENNI, Yann RENVOISÉ
- **Sujet** : conception d'un système d'aide à la décision (SAD) en radiologie à partir d'IRM cérébrales

## Résumé du projet
Ce projet implémente un **pipeline complet d'aide à la décision clinique** pour la classification de tumeurs cérébrales sur IRM.

Le système ne se limite pas à prédire une classe. Il fournit aussi :
- des **probabilités calibrées**,
- un **niveau de certitude**,
- une **recommandation clinique** avec priorité,
- une logique de **sécurité renforcée** contre les faux négatifs,
- des **rapports textuels automatisés** patients.

Le SAD est un outil d'assistance et **ne remplace pas le radiologue**.

## Contexte et objectifs pédagogiques
Le sujet impose de distinguer :
- **classification brute** (étiquette prédite),
- **décision clinique** (action recommandée selon risque et confiance).

Objectifs du module :
- implémenter un système multi-seuils,
- calibrer les probabilités (fiabilité décisionnelle),
- évaluer le modèle avec des métriques métier,
- produire un workflow de triage reproductible de bout en bout.

## Données
Dataset utilisé :
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Classes cibles :
- `glioma` : tumeur maligne (cas urgent)
- `meningioma` : tumeur bénigne (surveillance)
- `pituitary` : traitement spécialisé
- `notumor` : pas de tumeur (avec garde-fou de sécurité)

## Architecture fonctionnelle
Pipeline global :

`IRM -> Prétraitement -> Modèle (RegLog / MLP / CNN) -> Calibration -> Moteur de décision -> Rapport patient`

Composants implémentés :
- prétraitement image et extraction de features,
- modèles sklearn (régression logistique, MLP),
- modèle CNN PyTorch (`custom` ou `resnet18`),
- calibration probabiliste (`sigmoid` / `isotonic` + température CNN),
- moteur de décision clinique à seuils,
- estimation d'incertitude par MC Dropout,
- évaluation métier et sélection du meilleur modèle.

## Règles décisionnelles (métier)
Seuils appliqués :
- `max_prob >= 0.85` : diagnostic automatique validé
- `0.65 <= max_prob < 0.85` : révision recommandée
- `0.50 <= max_prob < 0.65` : cas incertain
- `max_prob < 0.50` : double lecture obligatoire

Règle de sécurité asymétrique :
- si classe prédite = `notumor` et `max_prob < 0.95` -> **vérification humaine obligatoire**.

## Critères d'évaluation (cahier des charges)
Métriques techniques et métier :
- accuracy globale,
- matrice de confusion et rapport de classification,
- accuracy par tranches de confiance,
- taux de couverture automatique,
- faux négatifs (FN), faux positifs (FP),
- coût métier :

`Cout_total = (FN * 1000) + (FP * 100) + (Revision * 50)`

Critère clé du sujet :
- **Accuracy > 95% quand confiance > 0.85**.

## Fonctions demandées dans l'énoncé
Fonctions implémentées dans le dépôt :
- `predire_avec_confiance(...)` -> `src/decision_engine.py`
- `generer_recommandation(...)` -> `src/decision_engine.py`
- `calculer_incertitude_mc_dropout(...)` -> `src/uncertainty.py`
- `creer_rapport_decision(...)` -> `src/reporting.py`

## Structure du dépôt
```text
Projet_SAD/
├── data/                          # Dataset (non versionné)
│   ├── Training/
│   └── Testing/
├── notebooks/
│   └── SAD_brain_tumor.ipynb      # Notebook principal
├── reports/
│   └── sample_reports.txt         # 20 rapports générés
├── src/
│   ├── calibration.py
│   ├── decision_engine.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── reporting.py
│   └── uncertainty.py
├── requirements.txt
└── README.md
```

## Installation
### 1) Créer et activer l'environnement
```bash
cd /chemin/vers/Projet_SAD
python -m venv venv
source venv/bin/activate
```

Sous Windows :
```bash
venv\Scripts\activate
```

### 2) Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Ajouter un kernel Jupyter (recommandé)
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name sad_v2 --display-name "Python (SAD v2)"
```

## Téléchargement du dataset (Kaggle CLI)
Pré-requis : avoir créé `~/.kaggle/kaggle.json` depuis son compte Kaggle.

```bash
python -m pip install --upgrade kaggle
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p .
unzip -o brain-tumor-mri-dataset.zip -d data
```

Arborescence attendue :
```text
data/
  Training/
    glioma/ meningioma/ notumor/ pituitary/
  Testing/
    glioma/ meningioma/ notumor/ pituitary/
```

## Exécution du notebook
```bash
jupyter notebook
```
Ouvrir ensuite : `notebooks/SAD_brain_tumor.ipynb`

Paramètres principaux dans le notebook :
- `FAST_MODE = True` : exécution rapide (itérations courtes)
- `FAST_MODE = False` : entraînement complet (run long)
- `RUN_CNN = True` : active l'entraînement CNN

Détection device CNN (ordre robuste) :
- `cuda` si disponible,
- sinon `mps` (Apple Silicon),
- sinon `cpu`.

## Résultats obtenus (dernier run long validé)
Comparaison des candidats calibrés :
- `RegLog+PCA+Calibration`
- `MLP+PCA+Calibration`
- `CNN-resnet18+TempScaling`

Modèle retenu (critère métier) :
- **CNN-resnet18+TempScaling**

Indicateurs clés du run validé :
- Accuracy globale (test) : **0.9931**
- Accuracy haute confiance (`conf > 0.85`) : **0.9969**
- Couverture automatique : **0.9931**
- Faux négatifs : **0**
- Faux positifs : **2**
- Coût métier : **650**

Conclusion de conformité :
- le critère `Accuracy@0.85 >= 0.95` est **atteint**.

## Vérification des livrables attendus
Checklist projet :
- RegLog évalué : **OK**
- MLP évalué : **OK**
- CNN évalué : **OK**
- Comparaison des modèles : **OK**
- Modèle final retenu : **OK**
- Génération de 20 rapports : **OK**
- Critère `Accuracy@0.85 >= 0.95` : **OK**

Vérification des rapports générés :
```bash
grep -c "^Patient ID:" reports/sample_reports.txt
```
La commande doit retourner `20`.

## Reproductibilité
Choix pour stabiliser les résultats :
- seed globale fixée (`SEED = 42`),
- split de calibration contrôlé,
- sélection de calibration sur split interne (fit/eval) puis refit final,
- calibration CNN (temperature scaling) ajustée sur **validation** puis évaluée sur **test**,
- split train/validation CNN stratifié,
- `pretrained=True` en mode long pour `resnet18`,
- protocole identique d'évaluation sur le jeu de test.

Remarque : de légères variations peuvent apparaître selon machine, backend GPU et version des librairies.

## Limites et cadre éthique
- Ce SAD est un prototype académique.
- Les performances sur Kaggle ne garantissent pas une généralisation clinique réelle.
- Les faux négatifs restent le risque principal ; la règle `notumor < 0.95` sert de garde-fou.
- Toute utilisation réelle exigerait validation multicentrique, protocole clinique et supervision médicale.

## Références
- Sujet du module : `Projet_Machine_learning.pdf`
- Dataset : [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Auteurs
- Aïssa MEHENNI
- Yann RENVOISÉ
