"""
Zadaniem skryptu jest zbudowanie modelu rozpatrujący dopasowanie sugerowanego tytułu do odpowiedniego manuskyrptu
wykorzystując wyniki z 02_prepare_data.py
"""

# Importowane biblioteki
import os
import time
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżki do głównego folderu / miejsca z wyganami plikami / miejsce z dla danych utworzonych
DATA_DIR = r"C:\Marcin\Podyplomowka_AnalizaDanych\Project_journalRecommender\JournalRecommender\data"
INPUT_FILE = "prepared_data.csv"
OUTPUT_DATA_FILE = "ml_results_data.csv"
MODEL_BUNDLE_PATH = os.path.join(DATA_DIR, "model_bundle.pkl")

def run_machine_learning_analysis():
    custom_vocab = [
        'GNSS', 'gnss', 'Global Navigation Satellite System', 'global navigation satellite system', 'GPS', 'gps',
        'Galileo', 'galileo', 'GLONASS', 'BeiDou', 'BDS', 'PPP', 'ppp', 'precise point positioning', 'Precise Point Positioning',
        'ambiguity resolution', 'ambiguities', 'AR', 'PPP-AR', 'PPP-RTK', 'IPPP', 'multi-GNSS', 'external clock', 'external clocks',
        'internal clock', 'internal clocks', 'receiver clock', 'atomic clock', 'rubidium', 'cesium', 'hydrogen-maser', 'hydrogen maser',
        'link time', 'link-time', 'timing', 'time-transfer', 'time transfer', 'time-link', 'time link', 'frequency transfer',
        'frequency-transfer', 'IGS','common clock', 'ISB', 'ISBs', 'MDEV', 'ADEV', 'random-walk', 'ZTD', 'ZWD', 'ZHD', 'troposhere',
        'Troposhhere', 'coordinates', 'vertical', 'hight', 'positioning', 'inter-system biases', 'Galileo-only', 'GPS-only',
        'GLONASS-only', 'BeiDou-only', 'stochastic modeling', 'receiver stochastic modeling', 'Stochastic modeling of the receiver clock',
        'IGS receiver clock', 'multi-GNSS PPP', 'station coordinates', 'troposphere delays', 'receiver clock modeling'
    ]

    custom_vocab = list(dict.fromkeys([w.lower() for w in custom_vocab]))

    # Wczytanie danych
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    if not os.path.exists(file_path):
        print(f"[BŁĄD] Nie znaleziono pliku {file_path}")
        return
    df = pd.read_csv(file_path)

    # Przygotowanie danych
    df = df.dropna(subset=['abstract']).copy()
    if 'impact_factor' in df.columns:
        df['impact_factor'] = df['impact_factor'].fillna(df['impact_factor'].median())
    else:
        df['impact_factor'] = 0.0
    for col in ['is_open_access', 'title_word_count', 'abstract_word_count']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Filtrowanie tematyczne
    vocab_pattern = r'\b(' + '|'.join([re.escape(w) for w in custom_vocab]) + r')\b'
    mask = df['abstract'].astype(str).str.lower().str.contains(vocab_pattern, regex=True, na=False)
    df_use = df[mask].copy() if mask.any() else df.copy()
    print(f"[INFO] Analiza na {len(df_use)} artykułach.")

    # Reset indeksów, co zapewnia zgodność pozycji macierzy TF-IDF z indeksami df_use
    df_use.reset_index(drop=True, inplace=True)

    # TF-IDF
    tfidf = TfidfVectorizer(vocabulary=custom_vocab, stop_words='english', ngram_range=(1, 2), lowercase=True)
    X_tfidf = tfidf.fit_transform(df_use['abstract'].astype(str))
    print(f"[INFO] TF-IDF shape: {X_tfidf.shape}")

    # KMeans
    n_clusters = min(5, X_tfidf.shape[0]) if X_tfidf.shape[0] >= 2 else 1
    kmeans = None
    if n_clusters > 1:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_tfidf)
            print(f"[INFO] KMeans utworzono {n_clusters} klastrów.")
        except Exception as e:
            print(f"[WARNING] Błąd KMeans: {e}")
            clusters = np.zeros(X_tfidf.shape[0], dtype=int)
            kmeans = None
    else:
        clusters = np.zeros(X_tfidf.shape[0], dtype=int)

    df_use['cluster'] = clusters

    # Przygotowanie danych do modelu
    X_numeric = df_use[['is_open_access', 'title_word_count', 'abstract_word_count', 'cluster']].values
    X_combined = hstack([X_tfidf, X_numeric])
    y = df_use['impact_factor'].values

    if X_combined.shape[0] < 2:
        print("[BŁĄD] Za mało próbek do trenowania.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Modele do porównania
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        'SVR': SVR(kernel='rbf')
    }

    best_model = None
    best_r2 = -np.inf
    best_model_name = None
    results = {}

    print("\n[TRENING] Porównanie modeli:")
    for name, model in models.items():
        print(f"\n[TRENING] Model: {name}")
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"[ERROR] Nie udało się wytrenować {name}: {e}")
            continue

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            cv_mean = np.mean(cv_scores)
        except Exception:
            cv_mean = np.nan
        elapsed = time.time() - start_time

        print(f"  → R²: {r2:.4f}, RMSE: {rmse:.4f}, CV_mean_R²: {cv_mean if not np.isnan(cv_mean) else 'n/a'}, time: {elapsed:.2f}s")
        results[name] = {'model': model, 'r2': r2, 'rmse': rmse, 'cv_mean': cv_mean, 'time': elapsed}
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    if best_model is None:
        print("[BŁĄD] Żaden model nie został wybrany jako najlepszy.")
        return

    print(f"\n[Najlepszy model]: {best_model_name} (R² = {best_r2:.4f})")

    # Zapis model bundle
    try:
        bundle = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'tfidf_vectorizer': tfidf,
            'kmeans': kmeans,
            'custom_vocab': custom_vocab,
            'feature_info': {
                'tfidf_features': list(tfidf.get_feature_names_out()),
                'numeric_features': ['is_open_access', 'title_word_count', 'abstract_word_count', 'cluster']
            }
        }
        with open(MODEL_BUNDLE_PATH, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[INFO] Model bundle zapisany: {MODEL_BUNDLE_PATH}")
    except Exception as e:
        print(f"[WARNING] Nie udało się zapisać bundle: {e}")

    # Wykres 1: najczęściej występujące słowa
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        tfidf_features = list(tfidf.get_feature_names_out())
        n_tfidf = len(tfidf_features)
        if importances.shape[0] < n_tfidf:
            print("[WARNING] feature_importances_ ma mniejszy rozmiar niż liczba cech TF-IDF. Pomijam wykres ważności.")
        else:
            word_importance_df = pd.DataFrame({
                'feature': tfidf_features,
                'importance': importances[:n_tfidf]
            }).sort_values(by='importance', ascending=False).head(50)

            plt.figure(figsize=(12, 10))
            sns.barplot(x='importance', y='feature', data=word_importance_df, color='tab:purple', dodge=False)
            plt.title("Top Keywords Influencing Impact Factor")
            plt.xlabel("Importance")
            plt.ylabel("Keyword")
            plt.tight_layout()
            path1 = os.path.join(DATA_DIR, "top_keywords_importance.png")
            plt.savefig(path1)
            plt.close()
            print(f"[INFO] Wykres top  słów zapisany: {path1}")
    else:
        print("[INFO] Model nie ma feature_importances_, pomijam wykres słów.")

    # Wykres 2: Heatmapa rozkładu top 20 słów w czasopismach
    print("[INFO] Generowanie wykresu dystrybucji słów w czasopismach...")

    tfidf_features = list(tfidf.get_feature_names_out())
    journal_list = df_use['journal'].unique().tolist()
    journal_word_sums = pd.DataFrame(index=journal_list, columns=tfidf_features, data=0.0)

    for i, word in enumerate(tfidf_features):
        col = X_tfidf[:, i].toarray().ravel()
        for j in journal_list:
            mask_j = (df_use['journal'] == j).values
            if mask_j.sum() == 0:
                s = 0.0
            else:
                s = float(col[mask_j].sum())
            journal_word_sums.at[j, word] = s

    journal_word_sums = journal_word_sums.fillna(0.0).astype(float)

    global_word_sums = journal_word_sums.sum(axis=0).sort_values(ascending=False).head(20)
    top_words_global = global_word_sums.index.tolist()
    print(f"[DEBUG] Global top words: {top_words_global}")

    heatmap_data = journal_word_sums[top_words_global]
    heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    if heatmap_norm.shape[0] == 0 or heatmap_norm.shape[1] == 0:
        print("[WARNING] Brak danych do wygenerowania heatmapy (za mało journali lub słów).")
    else:
        plt.figure(figsize=(14, max(6, len(heatmap_norm)*0.3)))
        sns.heatmap(heatmap_norm, cmap="YlGnBu", linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Proportion of TF-IDF'})
        plt.title("Top 20 Keywords Distribution Across Journals (Normalized TF-IDF)")
        plt.xlabel("Keyword")
        plt.ylabel("Journal")
        plt.tight_layout()
        path2 = os.path.join(DATA_DIR, "keywords_distribution_journals.png")
        plt.savefig(path2)
        plt.close()
        print(f"[INFO] Wykres dystrybucji słów w czasopismach zapisany: {path2}")

    # Zapis df_use z predykcjami
    try:
        df_use = df_use.copy()
        df_use['predicted_impact_factor'] = best_model.predict(X_combined)
        output_data_path = os.path.join(DATA_DIR, OUTPUT_DATA_FILE)
        df_use.to_csv(output_data_path, index=False)
        print(f"[INFO] Dane z predykcjami zapisano do: {output_data_path}")
    except Exception as e:
        print(f"[WARNING] Nie udało się zapisać predykcji: {e}")

# Generowanie statystyk opisowych dla wybranych kolumn i zapis do pliku CSV
    try:
        stats = {}

        # status - rozkład liczebności (count per category)
        if 'status' in df_use.columns:
            stats['status_counts'] = df_use['status'].value_counts(dropna=False)

        # abstract - długość abstraktu (liczba słów) - opis statystyczny
        if 'abstract' in df_use.columns:
            abstract_word_counts = df_use['abstract'].astype(str).apply(lambda x: len(x.split()))
            stats['abstract_word_count_desc'] = abstract_word_counts.describe()

        # publication_date - rozkład lat (jeśli jest w formacie daty lub string)
        if 'publication_date' in df_use.columns:
            try:
                dates = pd.to_datetime(df_use['publication_date'], errors='coerce')
                stats['publication_year_counts'] = dates.dt.year.value_counts(dropna=True).sort_index()
            except Exception:
                pass

        # is_open_access - rozkład liczebności
        if 'is_open_access' in df_use.columns:
            stats['is_open_access_counts'] = df_use['is_open_access'].value_counts(dropna=False)

        # title_word_count - opis statystyczny
        if 'title_word_count' in df_use.columns:
            stats['title_word_count_desc'] = df_use['title_word_count'].describe()

        # abstract_word_count - opis statystyczny
        if 'abstract_word_count' in df_use.columns:
            stats['abstract_word_count_desc'] = df_use['abstract_word_count'].describe()

        # cluster - rozkład liczebności
        if 'cluster' in df_use.columns:
            stats['cluster_counts'] = df_use['cluster'].value_counts(dropna=False).sort_index()

        # Zapis statystyk do pliku CSV w formacie "nazwa_statystyki;wartość"
        stats_output_path = os.path.join(DATA_DIR, "ml_results_data_stats.csv")
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            for key, series in stats.items():
                f.write(f"### {key} ###\n")
                if isinstance(series, pd.Series):
                    series.to_csv(f, header=True)
                elif isinstance(series, pd.DataFrame):
                    series.to_csv(f)
                else:
                    f.write(str(series) + "\n")
                f.write("\n")
        print(f"[INFO] Statystyki opisowe zapisano do: {stats_output_path}")
    except Exception as e:
        print(f"[WARNING] Nie udało się wygenerować statystyk opisowych: {e}")

if __name__ == "__main__":
    run_machine_learning_analysis()