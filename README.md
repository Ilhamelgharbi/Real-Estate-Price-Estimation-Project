# 🏠 Estimateur Immobilier Intelligent - Maroc

Une application web interactive basée sur l'intelligence artificielle pour estimer les prix des appartements au Maroc.

## 🚀 Fonctionnalités

- **Prédiction de prix en temps réel** : Estimation instantanée basée sur les caractéristiques de l'appartement
- **Interface intuitive** : Interface web facile à utiliser avec Streamlit
- **Analyse comparative** : Comparaison avec les prix moyens par ville
- **Conseils d'amélioration** : Suggestions pour augmenter la valeur de votre bien
- **Visualisations interactives** : Graphiques et métriques détaillées

## 🛠️ Installation

### Prérequis
- Python 3.7 ou plus récent
- pip (gestionnaire de packages Python)

### Étapes d'installation

1. **Cloner ou télécharger le projet**
```bash
git clone <repository-url>
cd gemini
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Générer les fichiers de modèle** (si pas déjà fait)
```bash
python test3.py
```

4. **Lancer l'application Streamlit**
```bash
streamlit run streamlit_app.py
```

Ou double-cliquez sur `run_app.bat` sous Windows.

## 📊 Utilisation

### Interface Principal

1. **Paramètres de base** (sidebar):
   - Sélectionnez la ville
   - Définissez la surface en m²
   - Choisissez le nombre de chambres
   - Définissez le nombre de salles de bains
   - Nombre de salons

2. **Équipements** (sidebar):
   - Sélectionnez parmi 16 équipements disponibles
   - Ascenseur, parking, climatisation, etc.

3. **Estimation** :
   - Cliquez sur "🔮 Estimer le Prix"
   - Obtenez une estimation instantanée

### Résultats Affichés

- **Prix Estimé** : Prix en dirhams marocains (MAD)
- **Prix par m²** : Prix au mètre carré
- **Catégorie** : Classification du bien (Économique, Moyen, Élevé, Très Élevé)
- **Analyse Comparative** : Graphique comparatif avec les autres villes
- **Détails** : Résumé des caractéristiques
- **Conseils** : Suggestions pour améliorer la valeur

## 🔧 Fichiers Requis

L'application nécessite les fichiers suivants générés par `test3.py`:

- `best_real_estate_model.pkl` : Modèle d'IA entraîné
- `minmax_scaler.pkl` : Scaler pour la normalisation
- `label_encoders.pkl` : Encodeurs pour les variables catégorielles
- `feature_columns.pkl` : Liste des colonnes de features
- `city_stats.pkl` : Statistiques des villes

## 📈 Performance du Modèle

- **Algorithme** : Gradient Boosting Regressor optimisé
- **Précision** : R² = 0.71 en validation croisée
- **Données d'entraînement** : 1400+ appartements au Maroc
- **Features** : 20 caractéristiques sélectionnées automatiquement

## 🏙️ Villes Supportées

- Casablanca
- Rabat
- Marrakech
- Fès
- Tanger
- Agadir
- Kénitra
- Mohammedia
- Salé
- Temara
- Et plus...

## 🎯 Équipements Supportés

- Ascenseur
- Parking
- Climatisation
- Terrasse
- Balcon
- Piscine
- Sécurité
- Concierge
- Chauffage
- Cuisine équipée
- Jardin
- Vue mer
- Meublé
- Garage
- Cave
- Interphone

## 🔍 Fonctionnalités Avancées

### Calculs Automatiques
- Densité des pièces
- Efficacité spatiale
- Ratios chambres/salles de bains
- Interactions entre caractéristiques

### Catégorisation Intelligente
- Surface (Studio, Petit, Moyen, Grand, Très Grand)
- Niveau de luxe (Standard, Confortable, Luxe, Premium)
- Niveau de prix par ville

### Conseils Personnalisés
- Suggestions d'amélioration basées sur l'analyse
- Estimation de l'impact sur la valeur
- Priorités d'investissement

## 📱 Interface Responsive

L'application s'adapte automatiquement à différentes tailles d'écran :
- Desktop
- Tablet
- Mobile

## 🛡️ Sécurité et Confidentialité

- Aucune donnée personnelle n'est stockée
- Traitement local des données
- Pas de connexion internet requise après installation

## 🔄 Mise à Jour du Modèle

Pour mettre à jour le modèle avec de nouvelles données :

1. Ajoutez de nouveaux données à `appartements-data-db.csv`
2. Exécutez `python test3.py` pour réentraîner
3. Relancez l'application Streamlit

## 📞 Support

Pour toute question ou problème :
- Vérifiez que tous les fichiers .pkl sont présents
- Assurez-vous que Python 3.7+ est installé
- Vérifiez que toutes les dépendances sont installées

## 🌟 Fonctionnalités Futures

- [ ] Support de plus de types de biens (maisons, villas)
- [ ] Prédictions de tendances du marché
- [ ] Analyse de rentabilité locative
- [ ] Export des rapports PDF
- [ ] API REST pour intégration

## 📄 Licence

Ce projet est destiné à des fins éducatives et de démonstration.

---

**Note** : Les estimations sont basées sur des données historiques et doivent être utilisées comme guide. Pour des évaluations officielles, consultez un expert immobilier.
