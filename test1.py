# ✅ QUICK IMPROVED MODEL TEST
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 QUICK IMPROVED MODEL TEST")
print("=" * 40)

# ✅ Chargement et nettoyage avancé
df = pd.read_csv("appartements-data-db.csv")
print(f"📊 Dataset original: {df.shape}")

# Nettoyer les prix correctement
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).replace("DH", "").replace(" ", "").replace(",", "")
    import re
    price_clean = re.sub(r'[^\d.]', '', price_str)
    try:
        return float(price_clean)
    except:
        return np.nan

df["price"] = df["price"].apply(clean_price)

# Supprimer les lignes avec price ou surface_area manquants
df = df.dropna(subset=['price', 'surface_area'])
print(f"📉 Après suppression des NaN critiques: {df.shape}")

# Nettoyer surface_area
df['surface_area'] = pd.to_numeric(df['surface_area'], errors='coerce')
df = df.dropna(subset=['surface_area'])

# ✅ Feature Engineering amélioré
df["price_per_m2"] = df["price"] / df["surface_area"]

# Imputer les colonnes numériques
for col in ['nb_rooms', 'nb_baths', 'salon']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Features avancées
df["total_rooms"] = df["nb_rooms"] + df["salon"]
df["rooms_per_m2"] = df["nb_rooms"] / df["surface_area"]
df["space_efficiency"] = df["total_rooms"] / df["surface_area"]

# ✅ Gestion des villes
arabic_to_french = {
    "الدار البيضاء": "Casablanca", "فاس": "Fès", "طنجة": "Tanger",
    "مراكش": "Marrakech", "أكادير": "Agadir", "الرباط": "Rabat"
}
df["city_name"] = df["city_name"].replace(arabic_to_french).fillna("Unknown")

# Encoder les villes
le = LabelEncoder()
df["city_encoded"] = le.fit_transform(df["city_name"])

# ✅ Suppression des outliers avec IQR
def remove_outliers(data, columns, factor=1.5):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        before = len(data)
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        print(f"   {col}: {before - len(data)} outliers supprimés")
    return data

print("\n🎯 Suppression des outliers:")
df_clean = remove_outliers(df, ['price', 'surface_area', 'price_per_m2'])
print(f"📊 Dataset final: {df_clean.shape}")

# ✅ Préparation des features
features = ['surface_area', 'nb_rooms', 'nb_baths', 'salon', 'total_rooms', 
           'price_per_m2', 'rooms_per_m2', 'space_efficiency', 'city_encoded']

X = df_clean[features]
y = df_clean["price"]

print(f"\n📈 Stats du prix target:")
print(f"   Min: {y.min():,.0f} DH")
print(f"   Max: {y.max():,.0f} DH")
print(f"   Mean: {y.mean():,.0f} DH")
print(f"   Median: {y.median():,.0f} DH")

# ✅ Mise à l'échelle robuste
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ✅ Test de modèles optimisés
models = {
    "Ridge_Optimized": Ridge(alpha=10.0),
    "RandomForest_Optimized": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5, 
        min_samples_leaf=2, random_state=42
    ),
    "GradientBoosting_Optimized": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6, 
        random_state=42
    )
}

print(f"\n🤖 ENTRAÎNEMENT (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
print("=" * 50)

best_score = -np.inf
best_model = None
best_name = None

for name, model in models.items():
    print(f"\n🔧 {name}...")
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"   R² Train: {r2_train:.4f}")
    print(f"   R² Test:  {r2_test:.4f}")
    print(f"   RMSE:     {rmse:,.0f} DH")
    
    # Vérifier l'overfitting
    overfitting = r2_train - r2_test
    print(f"   Overfitting: {overfitting:.4f}")
    
    if r2_test > best_score:
        best_score = r2_test
        best_model = model
        best_name = name

print(f"\n🏆 MEILLEUR MODÈLE: {best_name}")
print(f"📈 R² Score: {best_score:.4f}")

if best_score > 0.7:
    print("✅ EXCELLENT! R² > 0.7")
elif best_score > 0.5:
    print("🟡 BON! R² > 0.5") 
else:
    print("🔴 À AMÉLIORER! R² < 0.5")

# ✅ Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 TOP 5 FEATURES IMPORTANTES:")
    for i, row in importance_df.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

# ✅ Analyse des prédictions
print(f"\n📊 ANALYSE DES PRÉDICTIONS:")
y_pred_final = best_model.predict(X_test)
errors = np.abs(y_test - y_pred_final)
relative_errors = errors / y_test * 100

print(f"   Erreur absolue médiane: {np.median(errors):,.0f} DH")
print(f"   Erreur relative médiane: {np.median(relative_errors):.1f}%")
print(f"   Prédictions dans ±20%: {(relative_errors <= 20).mean()*100:.1f}%")

# Sauvegarder
joblib.dump(best_model, "best_model_quick.pkl")
joblib.dump(scaler, "scaler_quick.pkl")
print(f"\n💾 Modèle sauvegardé: best_model_quick.pkl")

print("\n✅ ANALYSE TERMINÉE!")
