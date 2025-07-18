# -*- coding: utf-8 -*-
"""
Outil d'estimation immobilière basé sur le machine learning pour le marché marocain.

Ce script implémente les étapes suivantes :
1. Chargement des données.
2. Analyse exploratoire des données (EDA).
3. Prétraitement des données (nettoyage, transformation, gestion des valeurs manquantes et aberrantes, encodage, mise à l'échelle).
4. Sélection des variables explicatives.
5. Entraînement et évaluation de plusieurs modèles de régression.
6. Optimisation des hyperparamètres via GridSearchCV.
7. Sélection du meilleur modèle.
8. Sauvegarde du modèle entraîné.
9. Test du modèle final avec comparaison des prix réels et prédits.

Améliorations spécifiques basées sur les bonnes pratiques :
- Gestion plus robuste de l'extraction des équipements.
- Suppression des lignes avec des valeurs manquantes dans 'price' (variable cible).
- Capping des valeurs extrêmes pour 'nb_baths' et 'surface_area' avant la détection d'outliers par IQR.
- Utilisation de LabelEncoder pour 'city_name'.
- Suppression de 'title'.
- Gestion des outliers par CAPPING (IQR) au lieu de la suppression.
- Vérification et suppression de la redondance entre les variables explicatives sélectionnées.
- Validation croisée supplémentaire pour le modèle optimisé.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")
import joblib
import re
import warnings

# Supprimer les avertissements pour une meilleure lisibilité
warnings.filterwarnings('ignore')

print("Début du processus d'estimation immobilière...")

# --- 1. Chargement des données ---
print("\n--- 1. Chargement des données ---")
try:
    df = pd.read_csv('appartements-data-db.csv')
    print("Données chargées avec succès.")
except FileNotFoundError:
    print("Erreur : Le fichier 'appartements-data-db.csv' est introuvable.")
    print("Veuillez vous assurer que le fichier est dans le même répertoire que le script.")
    exit()

print("\nInformations initiales sur le DataFrame :")
df.info()
print("\nPremières lignes du DataFrame :")
print(df.head())

# --- 2. Analyse exploratoire des données (EDA) ---
print("\n--- 2. Analyse exploratoire des données (EDA) ---")

print("\nDimensions du jeu de données :", df.shape)

print("\nValeurs manquantes par colonne (avant nettoyage initial) :")
print(df.isnull().sum())

print("\nNombre de lignes dupliquées :", df.duplicated().sum())
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print("Doublons supprimés. Nouvelle dimension :", df.shape)

print("\nStatistiques descriptives des variables numériques (avant nettoyage) :")
print(df.describe())

# Visualisation des distributions (exemple pour 'price' et 'surface_area')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Assurez-vous que 'price' est un type numérique pour la visualisation
temp_price = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
temp_price = pd.to_numeric(temp_price, errors='coerce').dropna()
sns.histplot(temp_price, bins=50, kde=True)
plt.title('Distribution des prix (avant nettoyage)')
plt.xlabel('Prix')
plt.ylabel('Fréquence')

plt.subplot(1, 2, 2)
sns.histplot(df['surface_area'].dropna(), bins=50, kde=True)
plt.title('Distribution de la surface (avant nettoyage)')
plt.xlabel('Surface')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.show()


# --- 3. Prétraitement des données ---
print("\n--- 3. Prétraitement des données ---")

# Nettoyage & Transformation

# 1. Convertir la colonne 'price' en type float et GÉRER les NaN de la cible (SUPPRESSION)
print("\n1. Conversion de la colonne 'price' en float et suppression des NaN...")
if 'price' in df.columns:
    df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    print("Colonne 'price' convertie en float.")

    # Suppression des lignes avec NaN dans 'price' (variable cible)
    initial_price_nan_rows = df['price'].isnull().sum()
    if initial_price_nan_rows > 0:
        df.dropna(subset=['price'], inplace=True)
        print(f"{initial_price_nan_rows} lignes avec des valeurs manquantes dans 'price' ont été supprimées.")
    print(f"Nouvelle dimension du DataFrame après suppression des NaN de 'price': {df.shape}")
else:
    print("La colonne 'price' n'existe pas. Skipping price conversion and NaN handling.")
    exit("La colonne 'price' est essentielle pour ce modèle. Arrêt du script.")


# 2. Traitement de la colonne 'city_name' (uniformisation et gestion des NaN)
print("\n2. Traitement de la colonne 'city_name' (uniformisation et gestion des NaN)...")
if 'city_name' in df.columns:
    city_mapping  = {
    "الدار البيضاء": "Casablanca",
    "دار بوعزة": "Dar Bouazza",
    "الرباط": "Rabat",
    "مراكش": "Marrakech",
    "أصيلة": "Asilah",
    "بوسكورة": "Bouskoura",
    "القنيطرة": "Kénitra",
    "المحمدية": "Mohammedia",
    "أكادير": "Agadir",
    "تمارة الجديدة": "Tamesna",
    "سلا": "Salé",
    "حد سوالم": "Had Soualem",
    "تمارة": "Temara",
    "بن سليمان": "Benslimane",
    "طنجة": "Tanger",
    "بوزنيقة": "Bouznika",
    "مكناس": "Meknès",
    "فاس": "Fès",
    "الجديدة": "El Jadida",
    "المنصورية": "El Mansouria",
    "مرتيل": "Martil",
    "الفنيدق": "Fnideq",
    "تطوان": "Tétouan",
    "السعيدية": "Saidia",
    "النواصر": "Nouaceur",
    "تماريس": "Tamaris",
    "كابو نيكرو": "Cabo Negro",
    "سيدي علال البحراوي": "Sidi Allal El Bahraoui",
    "بني ملال": "Béni Mellal",
    "غير معروف": "Unknown",
    "الصويرة": "Essaouira",
    "المهدية": "Mehdia",
    "وجدة": "Oujda",
    "وادي لاو": "Oued Laou",
    "الدشيرة": "Dcheira",
    "سيدي رحال": "Sidi Rahal",
    "دروة": "Deroua",
    "عين عتيق": "Ain Attig",
    "آسفي": "Safi",
    "إنزكان": "Inzegan",
    "إفران": "Ifrane",
    "الداخلة": "Dakhla",
    "الدشيرة الجهادية": "Dcheïra El Jihadia",
    "تغازوت": "Taghazout",
    "سيدي بوكنادل": "Sidi Bouknadel",
    "الصخيرات": "Skhirat",
    "خريبكة": "Khouribga",
    "بركان": "Berkane",
    "مرس الخير": "Mers El Kheir",
    "برشيد": "Berrechid",
    "تيزنيت": "Tiznit",
    "أكادير ملول": "Agadir Melloul",
    "الناظور": "Nador",
    "المنزه": "El Menzeh",
    "بني أنصار": "Bni Ansar",
    "المضيق": "Mdiq",
    "تيط مليل": "Tit Mellil",
    "سوق أربعاء": "Souk El Arbaa",
    "بيوڭرى": "Biougra",
    "سطات": "Settat",
    "عين عودة": "Ain Aouda",
    "تازة": "Taza",
    "الخميسات": "Khemisset",
    "وادي زم": "Oued Zem",
    "صفرو": "Sefrou",
    "مرزوكة": "Merzouga",
    "الحاجب": "El Hajeb",
    "سلوان": "Selouane",
    "تاونات": "Taounate",
    "سيدي بنور": "Sidi Bennour",
    "القصيبة": "El Ksiba"
}
    df['city_name'] = df['city_name'].replace(city_mapping)
    print("Noms de villes uniformisés.")
    df['city_name'].fillna('Unknown', inplace=True)
    print("Valeurs manquantes dans 'city_name' remplacées par 'Unknown'.")
else:
    print("La colonne 'city_name' n'existe pas. Skipping city_name treatment.")


# 3. Extraire les équipements (equipment) dans des colonnes booléennes
print("\n3. Extraction des équipements dans des colonnes booléennes...")
if 'equipment' in df.columns:
    df['equipment'] = df['equipment'].fillna('')
    all_equipments = set()
    for eq_list_str in df['equipment'].str.split(','):
        if isinstance(eq_list_str, list):
            for eq in eq_list_str:
                cleaned_eq = eq.strip().lower()
                if cleaned_eq:
                    all_equipments.add(cleaned_eq)

    for eq in sorted(list(all_equipments)):
        if eq:
            df[f'equipment_{eq.replace(" ", "_")}'] = df['equipment'].apply(lambda x: 1 if re.search(r'\b' + re.escape(eq) + r'\b', x.lower()) else 0)
    print(f"Colonnes d'équipement créées. Nombre de nouvelles colonnes : {len(all_equipments)}")
else:
    print("La colonne 'equipment' n'existe pas. Skipping equipment extraction.")


# 4. Supprimer les colonnes inutiles (equipment, link, title)
print("\n4. Suppression des colonnes inutiles (equipment, link, title)...")
columns_to_drop_final = ['equipment', 'link', 'title']
df.drop(columns=[col for col in columns_to_drop_final if col in df.columns], inplace=True)
print(f"Colonnes {', '.join([col for col in columns_to_drop_final if col in df.columns])} supprimées.")


# 5. Gestion des valeurs manquantes (pour les colonnes numériques restantes, hors price)
print("\n5. Gestion des valeurs manquantes (pour les colonnes numériques restantes)...")
numerical_cols_for_imputation = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'price' in numerical_cols_for_imputation:
    numerical_cols_for_imputation.remove('price')

for col in numerical_cols_for_imputation:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Valeurs manquantes dans '{col}' (numérique) imputées par la médiane ({median_val}).")

print("\nNouvelles valeurs manquantes après imputation :")
print(df.isnull().sum().loc[lambda x: x > 0])


# 6. Détection et GESTION avancée des valeurs aberrantes (Outliers)
print("\n6. Détection et GESTION avancée des valeurs aberrantes (Outliers)...")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import scipy.stats as stats

# Fonction pour détecter les outliers avec multiple méthodes
def detect_outliers_multiple_methods(data, column):
    """
    Détecte les outliers avec plusieurs méthodes et retourne un consensus
    """
    outliers = {}
    
    # 1. Méthode IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers['IQR'] = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    # 2. Méthode Z-score (pour distributions normales)
    z_scores = np.abs(stats.zscore(data[column]))
    outliers['Z-score'] = z_scores > 3
    
    # 3. Méthode Z-score modifié (plus robuste)
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    outliers['Modified Z-score'] = np.abs(modified_z_scores) > 3.5
    
    # 4. Méthode des percentiles (plus conservatrice)
    p01 = data[column].quantile(0.01)
    p99 = data[column].quantile(0.99)
    outliers['Percentile'] = (data[column] < p01) | (data[column] > p99)
    
    return outliers

# Fonction pour traiter les outliers avec différentes stratégies
def handle_outliers_advanced(data, column, strategy='adaptive_capping'):
    """
    Gère les outliers avec différentes stratégies
    """
    original_data = data[column].copy()
    
    if strategy == 'adaptive_capping':
        # Capping adaptatif basé sur la distribution
        if stats.skew(data[column]) > 1:  # Distribution très asymétrique
            # Utiliser des percentiles plus conservateurs
            lower_cap = data[column].quantile(0.05)
            upper_cap = data[column].quantile(0.95)
        else:
            # Utiliser IQR standard
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - 1.5 * IQR
            upper_cap = Q3 + 1.5 * IQR
        
        data[column] = np.clip(data[column], lower_cap, upper_cap)
        
    elif strategy == 'log_transformation':
        # Transformation logarithmique pour réduire l'impact des outliers
        if (data[column] > 0).all():
            data[column] = np.log1p(data[column])
        
    elif strategy == 'winsorization':
        # Winsorization (remplacer les valeurs extrêmes par des percentiles)
        lower_percentile = data[column].quantile(0.05)
        upper_percentile = data[column].quantile(0.95)
        data[column] = np.where(data[column] < lower_percentile, lower_percentile, data[column])
        data[column] = np.where(data[column] > upper_percentile, upper_percentile, data[column])
        
    elif strategy == 'robust_scaling':
        # Utiliser RobustScaler qui est moins sensible aux outliers
        scaler = RobustScaler()
        data[column] = scaler.fit_transform(data[[column]]).flatten()
        
    elif strategy == 'isolation_forest':
        # Utiliser Isolation Forest pour détecter et traiter les outliers
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(data[[column]])
        # Remplacer les outliers par la médiane
        median_val = data[column].median()
        data[column] = np.where(outliers == -1, median_val, data[column])
    
    return data[column], original_data

# Analyse des outliers par colonne
print("\n=== Analyse détaillée des outliers ===")
columns_to_analyze = ['price', 'surface_area', 'salon', 'nb_rooms', 'nb_baths']

outlier_summary = {}
for col in columns_to_analyze:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        print(f"\n--- Analyse des outliers pour '{col}' ---")
        
        # Statistiques descriptives
        print(f"Statistiques pour {col}:")
        print(f"  Moyenne: {df[col].mean():.2f}")
        print(f"  Médiane: {df[col].median():.2f}")
        print(f"  Écart-type: {df[col].std():.2f}")
        print(f"  Asymétrie (skewness): {stats.skew(df[col]):.2f}")
        print(f"  Aplatissement (kurtosis): {stats.kurtosis(df[col]):.2f}")
        
        # Détection des outliers avec multiple méthodes
        outliers_detected = detect_outliers_multiple_methods(df, col)
        
        outlier_counts = {}
        for method, outlier_mask in outliers_detected.items():
            outlier_count = outlier_mask.sum()
            outlier_counts[method] = outlier_count
            print(f"  {method}: {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
        
        outlier_summary[col] = outlier_counts

# Traitement adaptatif des outliers
print("\n=== Traitement adaptatif des outliers ===")

# Stratégies spécifiques par colonne
strategies = {
    'price': 'adaptive_capping',  # Pour les prix, capping adaptatif
    'surface_area': 'winsorization',  # Pour les surfaces, winsorization
    'salon': 'adaptive_capping',  # Pour les salons, capping adaptatif
    'nb_rooms': 'adaptive_capping',  # Pour les chambres, capping adaptatif
    'nb_baths': 'adaptive_capping'  # Pour les salles de bains, capping adaptatif
}

# Capping initial des valeurs extrêmes évidentes (erreurs de saisie)
print("\nCapping initial des valeurs extrêmes évidentes...")
if 'nb_baths' in df.columns:
    initial_nb_baths_outliers = df[df['nb_baths'] > 6].shape[0]
    df['nb_baths'] = np.where(df['nb_baths'] > 6, 6, df['nb_baths'])
    if initial_nb_baths_outliers > 0:
        print(f"Capping de {initial_nb_baths_outliers} valeurs extrêmes dans 'nb_baths' à 6.")

if 'surface_area' in df.columns:
    initial_surface_outliers = df[df['surface_area'] > 1000].shape[0]
    df['surface_area'] = np.where(df['surface_area'] > 1000, 1000, df['surface_area'])
    if initial_surface_outliers > 0:
        print(f"Capping de {initial_surface_outliers} valeurs extrêmes dans 'surface_area' à 1000.")

# Traitement adaptatif pour chaque colonne
for col in columns_to_analyze:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        strategy = strategies.get(col, 'adaptive_capping')
        
        print(f"\nTraitement des outliers pour '{col}' avec stratégie '{strategy}':")
        
        # Statistiques avant traitement
        before_stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
        
        # Appliquer le traitement
        df[col], original_values = handle_outliers_advanced(df, col, strategy)
        
        # Statistiques après traitement
        after_stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
        
        # Calculer les changements
        values_changed = (original_values != df[col]).sum()
        
        print(f"  Valeurs modifiées: {values_changed} ({values_changed/len(df)*100:.1f}%)")
        print(f"  Avant - Moyenne: {before_stats['mean']:.2f}, Médiane: {before_stats['median']:.2f}")
        print(f"  Après - Moyenne: {after_stats['mean']:.2f}, Médiane: {after_stats['median']:.2f}")
        print(f"  Réduction de l'écart-type: {((before_stats['std'] - after_stats['std'])/before_stats['std']*100):.1f}%")

# Créer une feature pour marquer les observations qui avaient des outliers
print("\n=== Création de features d'outliers ===")
for col in columns_to_analyze:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        # Détecter les outliers avec IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Créer une feature binaire pour les outliers
        df[f'{col}_was_outlier'] = ((df[col] <= lower_bound) | (df[col] >= upper_bound)).astype(int)
        
        outlier_count = df[f'{col}_was_outlier'].sum()
        if outlier_count > 0:
            print(f"Feature '{col}_was_outlier' créée: {outlier_count} observations marquées comme outliers")

# Visualiser l'effet du traitement des outliers
def plot_outlier_treatment_effects(df, columns_to_plot=['price', 'surface_area']):
    """
    Visualise l'effet du traitement des outliers sur les distributions
    """
    print("\n=== Visualisation des effets du traitement des outliers ===")
    
    fig, axes = plt.subplots(2, len(columns_to_plot), figsize=(15, 10))
    if len(columns_to_plot) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(columns_to_plot):
        if col in df.columns:
            # Calculer les statistiques actuelles (après traitement)
            current_mean = df[col].mean()
            current_median = df[col].median()
            current_std = df[col].std()
            current_min = df[col].min()
            current_max = df[col].max()
            
            # Histogramme avec distribution après traitement
            axes[0, i].hist(df[col], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            
            # Ajouter les lignes statistiques importantes
            axes[0, i].axvline(current_mean, color='red', linestyle='-', linewidth=2, label=f'Moyenne: {current_mean:.0f}')
            axes[0, i].axvline(current_median, color='green', linestyle='-', linewidth=2, label=f'Médiane: {current_median:.0f}')
            axes[0, i].axvline(current_mean + current_std, color='orange', linestyle='--', alpha=0.7, label=f'±1 Écart-type')
            axes[0, i].axvline(current_mean - current_std, color='orange', linestyle='--', alpha=0.7)
            
            axes[0, i].set_title(f'Distribution de {col} (après traitement des outliers)')
            axes[0, i].set_xlabel(f'{col} (Min: {current_min:.0f}, Max: {current_max:.0f})')
            axes[0, i].set_ylabel('Fréquence')
            axes[0, i].legend(loc='upper right')
            axes[0, i].grid(True, alpha=0.3)
            
            # Box plot avec informations détaillées
            bp = axes[1, i].boxplot(df[col], vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Ajouter les statistiques sur le box plot
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            axes[1, i].text(1.1, Q1, f'Q1: {Q1:.0f}', transform=axes[1, i].transData)
            axes[1, i].text(1.1, current_median, f'Médiane: {current_median:.0f}', transform=axes[1, i].transData)
            axes[1, i].text(1.1, Q3, f'Q3: {Q3:.0f}', transform=axes[1, i].transData)
            axes[1, i].text(1.1, current_mean, f'Moyenne: {current_mean:.0f}', transform=axes[1, i].transData, color='red')
            
            axes[1, i].set_title(f'Box plot de {col} (après traitement)\nIQR: {IQR:.0f}, Std: {current_std:.0f}')
            axes[1, i].set_ylabel(col)
            axes[1, i].grid(True, alpha=0.3)
            
            # Calculer le nombre d'observations qui étaient des outliers
            outlier_feature = f'{col}_was_outlier'
            if outlier_feature in df.columns:
                outlier_count = df[outlier_feature].sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                axes[1, i].text(0.5, 0.95, f'Outliers traités: {outlier_count} ({outlier_percentage:.1f}%)', 
                               transform=axes[1, i].transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Fonction pour créer un aperçu complet des effets du traitement des outliers
def create_outlier_treatment_summary(df, columns_to_analyze=['price', 'surface_area']):
    """
    Crée un résumé détaillé des effets du traitement des outliers
    """
    print("\n=== Résumé des effets du traitement des outliers ===")
    
    summary_data = []
    
    for col in columns_to_analyze:
        if col in df.columns:
            # Statistiques actuelles
            current_stats = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
            
            # Compter les outliers traités
            outlier_feature = f'{col}_was_outlier'
            outliers_treated = df[outlier_feature].sum() if outlier_feature in df.columns else 0
            outlier_percentage = (outliers_treated / len(df)) * 100
            
            summary_data.append({
                'Variable': col,
                'Outliers traités': outliers_treated,
                'Pourcentage outliers': f'{outlier_percentage:.1f}%',
                'Moyenne': f'{current_stats["mean"]:.0f}',
                'Médiane': f'{current_stats["median"]:.0f}',
                'Écart-type': f'{current_stats["std"]:.0f}',
                'Min': f'{current_stats["min"]:.0f}',
                'Max': f'{current_stats["max"]:.0f}',
                'IQR': f'{current_stats["iqr"]:.0f}'
            })
    
    # Créer un DataFrame pour l'affichage
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return summary_df

# Créer le résumé des effets du traitement des outliers
outlier_summary = create_outlier_treatment_summary(df, ['price', 'surface_area', 'salon', 'nb_rooms', 'nb_baths'])

# Appeler la fonction de visualisation
plot_outlier_treatment_effects(df, ['price', 'surface_area'])

print("\nDimensions du DataFrame après gestion avancée des outliers :", df.shape)


# 7. Encodage avancé des variables catégorielles (Target Encoding + LabelEncoder)
print("\n7. Encodage avancé des variables catégorielles (Target Encoding + LabelEncoder)...")
label_encoders = {} # Initialiser le dictionnaire pour sauvegarder les encodeurs
if 'city_name' in df.columns:
    # Calculer les statistiques par ville avant l'encodage
    city_stats = df.groupby('city_name')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
    city_stats.columns = ['city_name', 'city_price_mean', 'city_price_median', 'city_price_std', 'city_count']
    
    # Remplacer les valeurs NaN dans std par 0
    city_stats['city_price_std'] = city_stats['city_price_std'].fillna(0)
    
    # Créer un encodage target plus robuste avec régularisation
    # Utiliser une moyenne pondérée entre la moyenne de la ville et la moyenne globale
    global_mean = df['price'].mean()
    min_samples = 5  # Minimum d'échantillons pour faire confiance à la moyenne locale
    
    def target_encode_city(city_name):
        city_data = city_stats[city_stats['city_name'] == city_name]
        if len(city_data) == 0:
            return global_mean
        
        city_mean = city_data['city_price_mean'].iloc[0]
        city_count = city_data['city_count'].iloc[0]
        
        # Régularisation: plus on a d'échantillons, plus on fait confiance à la moyenne locale
        weight = city_count / (city_count + min_samples)
        return weight * city_mean + (1 - weight) * global_mean
    
    # Appliquer l'encodage target
    df['city_target_encoded'] = df['city_name'].apply(target_encode_city)
    print("Feature 'city_target_encoded' créée avec Target Encoding régularisé.")
    
    # Créer des features supplémentaires basées sur les villes
    city_stats_dict = city_stats.set_index('city_name').to_dict()
    df['city_price_std'] = df['city_name'].map(city_stats_dict['city_price_std'])
    df['city_sample_count'] = df['city_name'].map(city_stats_dict['city_count'])
    
    # Catégoriser les villes par leur niveau de prix
    price_percentiles = city_stats['city_price_mean'].quantile([0.25, 0.5, 0.75])
    def categorize_city_price_level(city_name):
        city_data = city_stats[city_stats['city_name'] == city_name]
        if len(city_data) == 0:
            return 1  # Moyenne
        
        city_mean = city_data['city_price_mean'].iloc[0]
        if city_mean <= price_percentiles[0.25]:
            return 0  # Économique
        elif city_mean <= price_percentiles[0.5]:
            return 1  # Moyenne
        elif city_mean <= price_percentiles[0.75]:
            return 2  # Chère
        else:
            return 3  # Très chère
    
    df['city_price_level'] = df['city_name'].apply(categorize_city_price_level)
    print("Feature 'city_price_level' créée (0=Économique, 1=Moyenne, 2=Chère, 3=Très chère).")
    
    # Garder aussi l'encodage LabelEncoder pour compatibilité
    le = LabelEncoder()
    df['city_name_encoded'] = le.fit_transform(df['city_name'])
    label_encoders['city_name'] = le # Sauvegarder l'encodeur
    print(f"Colonne 'city_name' encodée avec LabelEncoder (backup).")
    
    # Sauvegarder les informations d'encodage
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(city_stats, 'city_stats.pkl')
    print("Encodeurs et statistiques des villes sauvegardés.")
else:
    print("La colonne 'city_name' n'est pas présente pour l'encodage.")


# 8. Feature Engineering Amélioré (Ingénierie des caractéristiques)
print("\n8. Feature Engineering Amélioré (Ingénierie des caractéristiques)...")

# Créer des features dérivées avant la normalisation
if 'surface_area' in df.columns and 'price' in df.columns:
    # Prix par mètre carré (très important pour l'immobilier)
    df['price_per_sqm'] = df['price'] / (df['surface_area'] + 1e-6)  # Éviter division par zéro
    print("Feature 'price_per_sqm' créée.")
    
    # Catégories de prix par m²
    df['price_per_sqm_category'] = pd.cut(df['price_per_sqm'], bins=5, labels=['Très_bon_marché', 'Bon_marché', 'Moyen', 'Cher', 'Très_cher'])
    df['price_per_sqm_category'] = df['price_per_sqm_category'].astype('category').cat.codes
    print("Feature 'price_per_sqm_category' créée.")

if 'nb_rooms' in df.columns and 'nb_baths' in df.columns:
    # Ratio chambres/salles de bains
    df['rooms_to_baths_ratio'] = df['nb_rooms'] / (df['nb_baths'] + 1e-6)
    print("Feature 'rooms_to_baths_ratio' créée.")
    
    # Indicateur de confort (plus de salles de bains = plus de confort)
    df['comfort_ratio'] = df['nb_baths'] / (df['nb_rooms'] + 1e-6)
    print("Feature 'comfort_ratio' créée.")

if 'nb_rooms' in df.columns and 'nb_baths' in df.columns and 'salon' in df.columns:
    # Nombre total de pièces
    df['total_rooms'] = df['nb_rooms'] + df['nb_baths'] + df['salon']
    print("Feature 'total_rooms' créée.")

if 'surface_area' in df.columns and 'nb_rooms' in df.columns:
    # Surface par chambre
    df['surface_per_room'] = df['surface_area'] / (df['nb_rooms'] + 1e-6)
    print("Feature 'surface_per_room' créée.")

if 'surface_area' in df.columns:
    # Catégories de surface améliorées
    df['surface_category'] = pd.cut(df['surface_area'], 
                                  bins=[0, 50, 80, 120, 200, float('inf')], 
                                  labels=['Studio', 'Petit', 'Moyen', 'Grand', 'Très_grand'])
    df['surface_category'] = df['surface_category'].astype('category').cat.codes
    print("Feature 'surface_category' créée.")
    
    # Indicateur de taille
    df['is_spacious'] = (df['surface_area'] > df['surface_area'].quantile(0.8)).astype(int)
    df['is_compact'] = (df['surface_area'] < df['surface_area'].quantile(0.2)).astype(int)
    print("Features 'is_spacious' et 'is_compact' créées.")

# Améliorer les features d'équipements
equipment_cols = [col for col in df.columns if col.startswith('equipment_')]
if len(equipment_cols) > 0:
    # Nombre total d'équipements
    df['total_equipment'] = df[equipment_cols].sum(axis=1)
    print("Feature 'total_equipment' créée.")
    
    # Features spécifiques pour différents types d'équipements
    luxury_equipment = ['equipment_ascenseur', 'equipment_parking', 'equipment_climatisation', 
                       'equipment_terrasse', 'equipment_balcon', 'equipment_piscine']
    luxury_cols = [col for col in luxury_equipment if col in df.columns]
    if luxury_cols:
        df['luxury_equipment_count'] = df[luxury_cols].sum(axis=1)
        df['has_luxury_equipment'] = (df['luxury_equipment_count'] > 0).astype(int)
        print("Features 'luxury_equipment_count' et 'has_luxury_equipment' créées.")
    
    # Équipements de sécurité
    security_cols = [col for col in equipment_cols if 'sécurité' in col or 'concierge' in col]
    if security_cols:
        df['security_equipment'] = df[security_cols].sum(axis=1)
        df['has_security'] = (df['security_equipment'] > 0).astype(int)
        print("Features 'security_equipment' et 'has_security' créées.")
    
    # Équipements de confort
    comfort_equipment = ['equipment_climatisation', 'equipment_chauffage', 'equipment_cuisine_équipée']
    comfort_cols = [col for col in comfort_equipment if col in df.columns]
    if comfort_cols:
        df['comfort_equipment_count'] = df[comfort_cols].sum(axis=1)
        df['high_comfort'] = (df['comfort_equipment_count'] >= 2).astype(int)
        print("Features 'comfort_equipment_count' et 'high_comfort' créées.")

# Créer des features polynomiales pour les variables importantes
if 'surface_area' in df.columns:
    df['surface_area_squared'] = df['surface_area'] ** 2
    df['surface_area_sqrt'] = np.sqrt(df['surface_area'])
    df['surface_area_log'] = np.log1p(df['surface_area'])
    print("Features polynomiales pour 'surface_area' créées.")

if 'nb_rooms' in df.columns:
    df['nb_rooms_squared'] = df['nb_rooms'] ** 2
    print("Feature 'nb_rooms_squared' créée.")

# Créer des features d'interaction importantes
if 'surface_area' in df.columns and 'luxury_equipment_count' in df.columns:
    df['surface_luxury_interaction'] = df['surface_area'] * df['luxury_equipment_count']
    print("Feature 'surface_luxury_interaction' créée.")

if 'total_rooms' in df.columns and 'surface_area' in df.columns:
    df['rooms_surface_interaction'] = df['total_rooms'] * df['surface_area']
    print("Feature 'rooms_surface_interaction' créée.")

# Créer des features de densité et d'efficacité
if 'surface_area' in df.columns and 'total_rooms' in df.columns:
    df['room_density'] = df['total_rooms'] / (df['surface_area'] + 1e-6)
    df['space_efficiency'] = df['surface_area'] / (df['total_rooms'] + 1e-6)
    print("Features 'room_density' et 'space_efficiency' créées.")

# Créer des features basées sur des seuils intelligents
if 'surface_area' in df.columns:
    df['is_large_apartment'] = (df['surface_area'] > df['surface_area'].quantile(0.75)).astype(int)
    print("Feature 'is_large_apartment' créée.")

if 'nb_rooms' in df.columns:
    df['is_multi_room'] = (df['nb_rooms'] >= 3).astype(int)
    df['is_family_size'] = (df['nb_rooms'] >= 4).astype(int)
    print("Features 'is_multi_room' et 'is_family_size' créées.")

if 'luxury_equipment_count' in df.columns:
    df['is_luxury'] = (df['luxury_equipment_count'] >= 2).astype(int)
    df['is_premium'] = (df['luxury_equipment_count'] >= 3).astype(int)
    print("Features 'is_luxury' et 'is_premium' créées.")

# Créer des features de ratios avancés
if 'nb_baths' in df.columns and 'total_rooms' in df.columns:
    df['bath_to_total_rooms_ratio'] = df['nb_baths'] / (df['total_rooms'] + 1e-6)
    print("Feature 'bath_to_total_rooms_ratio' créée.")

# Créer des features de combinaisons d'équipements populaires
common_equipment_combos = [
    ('equipment_parking', 'equipment_ascenseur'),
    ('equipment_terrasse', 'equipment_balcon'),
    ('equipment_parking', 'equipment_sécurité'),
    ('equipment_climatisation', 'equipment_ascenseur')
]

for eq1, eq2 in common_equipment_combos:
    if eq1 in df.columns and eq2 in df.columns:
        combo_name = f"{eq1.replace('equipment_', '')}_and_{eq2.replace('equipment_', '')}"
        df[combo_name] = df[eq1] & df[eq2]
        print(f"Feature '{combo_name}' créée.")

# Créer des features basées sur le target encoding des villes
if 'city_target_encoded' in df.columns and 'surface_area' in df.columns:
    df['city_surface_interaction'] = df['city_target_encoded'] * df['surface_area']
    print("Feature 'city_surface_interaction' créée.")

if 'city_target_encoded' in df.columns and 'luxury_equipment_count' in df.columns:
    df['city_luxury_interaction'] = df['city_target_encoded'] * df['luxury_equipment_count']
    print("Feature 'city_luxury_interaction' créée.")

print(f"Nombre total de features après feature engineering amélioré: {df.shape[1]}")

# 9. Mise à l'échelle des variables numériques
print("\n9. Mise à l'échelle des variables numériques...")
numerical_cols_after_encoding = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'price' in numerical_cols_after_encoding:
    numerical_cols_after_encoding.remove('price')

# Exclure aussi price_per_sqm car elle est dérivée du prix
price_related_cols = ['price_per_sqm', 'city_price_mean']
cols_to_scale = [col for col in numerical_cols_after_encoding if col not in price_related_cols]

scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
print("Variables numériques mises à l'échelle avec MinMaxScaler.")

# Normaliser séparément les features dérivées du prix
if 'price_per_sqm' in df.columns:
    price_per_sqm_scaler = MinMaxScaler()
    df[['price_per_sqm']] = price_per_sqm_scaler.fit_transform(df[['price_per_sqm']])
    print("Feature 'price_per_sqm' normalisée séparément.")

if 'city_price_mean' in df.columns:
    city_price_scaler = MinMaxScaler()
    df[['city_price_mean']] = city_price_scaler.fit_transform(df[['city_price_mean']])
    print("Feature 'city_price_mean' normalisée séparément.")


# --- 4. Sélection avancée des variables explicatives ---
print("\n--- 4. Sélection avancée des variables explicatives ---")

# Supprimer la colonne city_name originale (string) avant le calcul de corrélation
if 'city_name' in df.columns:
    df = df.drop('city_name', axis=1)
    print("Colonne 'city_name' (string) supprimée avant calcul de corrélation.")

# Calcul de la matrice de corrélation
correlation_matrix = df.corr()

# Exclure les features dérivées du prix de la sélection automatique
price_derived_features = ['price_per_sqm', 'price_per_sqm_category', 'city_price_mean', 'city_target_encoded']
features_to_exclude = ['price'] + price_derived_features

print("=== ÉTAPE 1: Sélection basée sur la corrélation ===")
# Sélectionner les variables corrélées au prix avec un seuil plus élevé
if 'price' in correlation_matrix.columns:
    price_correlations = correlation_matrix['price'].abs().sort_values(ascending=False)
    
    # Utiliser un seuil plus élevé pour une meilleure sélection
    correlation_threshold = 0.2  # Augmenté de 0.15 à 0.2
    selected_features = price_correlations[price_correlations > correlation_threshold].index.tolist()
    
    # Exclure les features dérivées du prix et le prix lui-même
    selected_features = [f for f in selected_features if f not in features_to_exclude]
    
    print(f"Variables avec corrélation > {correlation_threshold} avec 'price': {len(selected_features)}")
    
    # Afficher les top features avec leurs corrélations
    print("\nTop 25 features les plus corrélées au prix:")
    for i, feature in enumerate(selected_features[:25]):
        corr_value = correlation_matrix.loc[feature, 'price']
        print(f"{i+1:2d}. {feature}: {corr_value:.4f}")
    
    if len(selected_features) > 25:
        print(f"... et {len(selected_features) - 25} autres features.")

print("\n=== ÉTAPE 2: Gestion avancée de la multicolinéarité ===")
if len(selected_features) > 1:
    selected_features_df = df[selected_features]
    inter_feature_corr = selected_features_df.corr().abs()
    
    # Fonction pour trouver les features hautement corrélées
    def find_correlated_features(corr_matrix, threshold=0.8):
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    corr_pairs.append((col1, col2, corr_value))
        return sorted(corr_pairs, key=lambda x: x[2], reverse=True)
    
    # Trouver les paires de features hautement corrélées
    correlated_pairs = find_correlated_features(inter_feature_corr, threshold=0.8)
    
    print(f"Paires de features avec corrélation > 0.8: {len(correlated_pairs)}")
    
    # Supprimer les features redondantes en gardant celle avec la meilleure corrélation au prix
    features_to_remove = set()
    for col1, col2, corr_value in correlated_pairs:
        if col1 not in features_to_remove and col2 not in features_to_remove:
            corr_col1_price = abs(correlation_matrix.loc[col1, 'price'])
            corr_col2_price = abs(correlation_matrix.loc[col2, 'price'])
            
            if corr_col1_price < corr_col2_price:
                features_to_remove.add(col1)
                print(f"  Suppression de '{col1}' (corr={corr_value:.3f} avec '{col2}', prix_corr={corr_col1_price:.3f})")
            else:
                features_to_remove.add(col2)
                print(f"  Suppression de '{col2}' (corr={corr_value:.3f} avec '{col1}', prix_corr={corr_col2_price:.3f})")
    
    # Appliquer les suppressions
    initial_count = len(selected_features)
    selected_features = [f for f in selected_features if f not in features_to_remove]
    
    print(f"\nFeatures après suppression de multicolinéarité: {initial_count} -> {len(selected_features)}")

print("\n=== ÉTAPE 3: Sélection finale basée sur l'importance ===")
# Limiter le nombre de features pour éviter l'overfitting
max_features = 20  # Limiter à 20 features maximum
if len(selected_features) > max_features:
    print(f"Limitation du nombre de features de {len(selected_features)} à {max_features}")
    
    # Garder les features avec la meilleure corrélation au prix
    feature_importance = [(f, abs(correlation_matrix.loc[f, 'price'])) for f in selected_features]
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    selected_features = [f[0] for f in feature_importance[:max_features]]
    
    print(f"Top {max_features} features sélectionnées:")
    for i, (feature, corr_value) in enumerate(feature_importance[:max_features]):
        print(f"{i+1:2d}. {feature}: {corr_value:.4f}")

print("\n=== ÉTAPE 4: Validation de la sélection ===")
# Vérifier la distribution des corrélations
correlations_with_price = [abs(correlation_matrix.loc[f, 'price']) for f in selected_features]
print(f"Corrélations avec le prix - Min: {min(correlations_with_price):.3f}, Max: {max(correlations_with_price):.3f}, Moyenne: {np.mean(correlations_with_price):.3f}")

# Vérifier les types de features sélectionnées
feature_types = {
    'surface': len([f for f in selected_features if 'surface' in f.lower()]),
    'equipment': len([f for f in selected_features if 'equipment' in f.lower()]),
    'room': len([f for f in selected_features if 'room' in f.lower() or 'bath' in f.lower()]),
    'city': len([f for f in selected_features if 'city' in f.lower()]),
    'interaction': len([f for f in selected_features if 'interaction' in f.lower()]),
    'other': len([f for f in selected_features if not any(keyword in f.lower() for keyword in ['surface', 'equipment', 'room', 'bath', 'city', 'interaction'])])
}

print(f"\nTypes de features sélectionnées:")
for feature_type, count in feature_types.items():
    if count > 0:
        print(f"  {feature_type}: {count} features")

print(f"\nFeatures finales sélectionnées ({len(selected_features)}):")
for i, feature in enumerate(selected_features):
    corr_value = correlation_matrix.loc[feature, 'price']
    print(f"{i+1:2d}. {feature}: {corr_value:.4f}")

else:
    print("La colonne 'price' n'est pas présente pour la sélection des variables.")
    selected_features = [col for col in df.columns if col not in features_to_exclude]
    print("Toutes les colonnes disponibles (sauf price et features dérivées) seront utilisées comme caractéristiques.")


# Séparation des données
print("\nSéparation des données en ensembles d'entraînement et de test...")
if 'price' in df.columns and len(selected_features) > 0:
    y = df["price"]
    X = df[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dimensions de l'ensemble d'entraînement : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Dimensions de l'ensemble de test : X_test={X_test.shape}, y_test={y_test.shape}")
else:
    print("Impossible de séparer les données car la colonne 'price' est manquante ou aucune caractéristique n'a été sélectionnée.")
    exit()

# --- 5. Entraînement et évaluation des modèles de régression ---
print("\n--- 5. Entraînement et évaluation des modèles de régression ---")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"\nEntraînement et évaluation de : {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MSE': mse, 'RMSE': mse, 'MAE': mae, 'R2': r2} # Corrected RMSE to MSE for consistency in results dict

    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")

    # Validation croisée
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  Scores R² de validation croisée (5-fold): {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Afficher un résumé des performances
print("\n--- Résumé des performances des modèles ---")
for name, metrics in results.items():
    print(f"\nModèle: {name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")

# --- 6. Optimisation des hyperparamètres (GridSearchCV pour plusieurs modèles) ---
print("\n--- 6. Optimisation des hyperparamètres (GridSearchCV pour plusieurs modèles) ---")

# Définir les grilles de paramètres pour les 4 modèles sélectionnés
param_grids = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

best_models = {}
best_scores = {}

for model_name, config in param_grids.items():
    print(f"\nOptimisation des hyperparamètres pour {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=config['model'], 
        param_grid=config['params'],
        cv=3, 
        n_jobs=-1, 
        verbose=1, 
        scoring='r2'
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")
    print(f"Meilleur score CV pour {model_name}: {grid_search.best_score_:.4f}")
    
    best_models[model_name] = grid_search.best_estimator_
    best_scores[model_name] = grid_search.best_score_
    
    # Évaluer sur l'ensemble de test
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Performance sur test - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
    
    # Ajouter aux résultats
    results[f'{model_name} (Optimisé)'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': test_rmse,
        'MAE': test_mae,
        'R2': test_r2
    }

# --- 7. Sélection du meilleur modèle ---
print("\n--- 7. Sélection du meilleur modèle ---")

best_model_name = None
best_r2_score = -np.inf

for name, metrics in results.items():
    if metrics['R2'] > best_r2_score:
        best_r2_score = metrics['R2']
        best_model_name = name

print(f"\nLe meilleur modèle est : {best_model_name} avec un R² de {best_r2_score:.2f}")

# Récupérer le meilleur modèle final
final_best_model = None
if best_model_name:
    # Vérifier si c'est un modèle optimisé
    if '(Optimisé)' in best_model_name:
        model_base_name = best_model_name.replace(' (Optimisé)', '')
        if model_base_name in best_models:
            final_best_model = best_models[model_base_name]
        else:
            # Fallback pour les anciens modèles optimisés
            for model_key in best_models:
                if model_key in best_model_name:
                    final_best_model = best_models[model_key]
                    break
    else:
        # Modèle de base
        if best_model_name in models:
            final_best_model = models[best_model_name]

if final_best_model is None:
    print("Aucun modèle n'a pu être sélectionné.")
    # Fallback: utiliser le premier modèle optimisé disponible
    if best_models:
        final_best_model = list(best_models.values())[0]
        print(f"Utilisation du modèle de fallback: {list(best_models.keys())[0]}")

# --- 6.5. Création d'un modèle d'ensemble ---
print("\n--- 6.5. Création d'un modèle d'ensemble ---")

if len(best_models) >= 2:
    # Sélectionner les 3 meilleurs modèles pour l'ensemble
    top_models = sorted(best_models.items(), key=lambda x: best_scores[x[0]], reverse=True)[:3]
    
    ensemble_estimators = [(name, model) for name, model in top_models]
    
    ensemble_model = VotingRegressor(estimators=ensemble_estimators)
    ensemble_model.fit(X_train, y_train)
    
    # Évaluer l'ensemble
    y_pred_ensemble = ensemble_model.predict(X_test)
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
    
    print(f"Modèles dans l'ensemble: {[name for name, _ in ensemble_estimators]}")
    print(f"Performance de l'ensemble - R²: {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.2f}, MAE: {ensemble_mae:.2f}")
    
    # Ajouter aux résultats
    results['Ensemble Model'] = {
        'MSE': mean_squared_error(y_test, y_pred_ensemble),
        'RMSE': ensemble_rmse,
        'MAE': ensemble_mae,
        'R2': ensemble_r2
    }
    
    # Vérifier si l'ensemble est meilleur
    if ensemble_r2 > best_r2_score:
        final_best_model = ensemble_model
        best_model_name = 'Ensemble Model'
        best_r2_score = ensemble_r2
        print(f"L'ensemble est maintenant le meilleur modèle avec R² = {ensemble_r2:.4f}")
else:
    print("Pas assez de modèles optimisés pour créer un ensemble.")

# --- Validation croisée supplémentaire pour le modèle optimisé (maintenant que final_best_model est défini) ---
print("\n--- Validation croisée supplémentaire pour le modèle optimisé ---")
if final_best_model:
    final_cv_scores = cross_val_score(final_best_model, X, y, cv=5, scoring='r2')
    print(f"Scores R² de validation croisée (5-fold) pour le modèle optimisé ({best_model_name}): {final_cv_scores.mean():.2f} (+/- {final_cv_scores.std():.2f})")
else:
    print("Aucun modèle optimisé à valider en cross-validation.")


# --- 8. Sauvegarde du modèle entraîné ---
if final_best_model:
    print("\n--- 8. Sauvegarde du modèle entraîné ---")
    model_filename = 'best_real_estate_model.pkl'
    joblib.dump(final_best_model, model_filename)
    print(f"Modèle '{best_model_name}' sauvegardé sous '{model_filename}'")

    # Sauvegarder le scaler et l'encodeur Label pour le déploiement
    joblib.dump(scaler, 'minmax_scaler.pkl')
    print("MinMaxScaler sauvegardé sous 'minmax_scaler.pkl'")
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("LabelEncoder pour 'city_name' sauvegardé sous 'label_encoders.pkl'")
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    print("Liste des colonnes de caractéristiques sauvegardée sous 'feature_columns.pkl'")

    print("\nProcessus d'estimation immobilière terminé avec succès.")
else:
    print("\nImpossible de sauvegarder le modèle car aucun modèle n'a été sélectionné.")


# --- 9. Test du modèle final : Comparaison des prix réels et prédits ---
print("\n--- 9. Test du modèle final : Comparaison des prix réels et prédits ---")

if final_best_model is not None:
    # Faire des prédictions sur l'ensemble de test
    y_pred_final = final_best_model.predict(X_test)

    print(f"\nComparaison des prix réels et prédits par le modèle final ({best_model_name}):")

    # Afficher les premières valeurs réelles et prédites
    comparison_df = pd.DataFrame({'Prix Réel': y_test, 'Prix Prédit': y_pred_final})
    print("\nPremières 10 comparaisons (Prix Réel vs Prix Prédit) :")
    print(comparison_df.head(10).round(2))

    # Graphique de dispersion : Prix Réel vs Prix Prédit
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_final, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ligne y=x
    plt.title('Prix Réels vs Prix Prédits')
    plt.xlabel('Prix Réel')
    plt.ylabel('Prix Prédit')
    plt.grid(True)
    plt.show()
    # Graphique des résidus : Prix Prédits vs Résidus (Erreurs)
    residuals = y_test - y_pred_final
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_final, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2) # Ligne de zéro résidu
    plt.title('Graphique des Résidus (Prix Prédits vs Erreurs)')
    plt.xlabel('Prix Prédit')
    plt.ylabel('Résidus (Erreur de Prédiction)')
    plt.grid(True)
    plt.show()

    print("\nAnalyse visuelle des prédictions terminée.")
else:
    print("Impossible de tester le modèle : aucun modèle final n'a été sélectionné ou entraîné.")
