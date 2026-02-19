"""
Zadaniem skryptu jest weryfikacja zbudowanego modelu z 03_machine_learining_model.py, przeprowadzając
dopasowanie sugerowanego modelu
"""

# Importowane biblioteki
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

# Ścieżki do głównego folderu / miejsca z wyganami plikami / miejsce z dla danych utworzonych
DATA_DIR = r"C:\Marcin\Podyplomowka_AnalizaDanych\Project_journalRecommender\JournalRecommender\data"
PREPARED_FILE = os.path.join(DATA_DIR, "prepared_data.csv")
MODEL_BUNDLE_PATH = os.path.join(DATA_DIR, "model_bundle.pkl")

# --- FUNKCJE pomocnicze ---
def load_bundle(path=MODEL_BUNDLE_PATH):
    """Wczytaj model bundle (zawiera tfidf_vectorizer i opcjonalnie model/kmeans)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono bundle z modelem: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def load_prepared_data(path=PREPARED_FILE):
    """Wczytaj prepared_data.csv i wstępnie przygotuj (wypełnienia, reset indeksów)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {path}")
    df = pd.read_csv(path)
    df['abstract'] = df.get('abstract', '').fillna('').astype(str)
    for col in ['impact_factor', 'is_open_access', 'title_word_count', 'abstract_word_count']:
        if col not in df.columns:
            if col == 'impact_factor':
                df[col] = 0.0
            else:
                df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    if 'journal' not in df.columns:
        raise ValueError("prepared_data.csv musi zawierać kolumnę 'journal'")
    df['journal'] = df['journal'].astype(str).str.strip()
    df.reset_index(drop=True, inplace=True)
    return df

def compute_journal_centroids(df, tfidf_vectorizer):
    """
    Oblicz centroid TF-IDF (średni wektor) dla każdego journal.
    Zwraca DataFrame z index=journal i kolumną 'centroid' (1D numpy array) i 'n_articles'.
    """
    abstracts = df['abstract'].astype(str).tolist()
    X = tfidf_vectorizer.transform(abstracts)
    journals = df['journal'].unique()
    centroids = []
    for j in journals:
        mask = (df['journal'] == j).values
        if mask.sum() == 0:
            continue
        rows = X[mask]
        centroid = rows.mean(axis=0)
        centroid = np.asarray(centroid).ravel()
        centroids.append({'journal': j, 'centroid': centroid, 'n_articles': int(mask.sum())})
    centroids_df = pd.DataFrame(centroids).set_index('journal')
    return centroids_df, X

def prepare_input_vector(title, abstract, tfidf_vectorizer):
    """Zwraca sparse 1 x n_features wektor TF-IDF dla tekstu (title + abstract)."""
    text = (title or '') + ' '
    if abstract:
        text += abstract
    text = text.strip()
    return tfidf_vectorizer.transform([text])

def verify_article(title, abstract=None, top_n=10):
    """
    Główna funkcja:
    - wczytuje bundle i dane
    - oblicza centroidy journali
    - liczy podobieństwo kosinusowe między wektorem wejścia a centroidami
    - zwraca DataFrame top_n najbliższych czasopism z informacjami
      (journal, n_articles, if, similarity)
    """
    bundle = load_bundle(MODEL_BUNDLE_PATH)
    tfidf = bundle.get('tfidf_vectorizer', None)

    if tfidf is None:
        raise RuntimeError("Bundle nie zawiera 'tfidf_vectorizer' - nie można kontynuować. Upewnij się, że model bundle został zapisany poprawnie.")

    df = load_prepared_data(PREPARED_FILE)

    centroids_df, X_tfidf = compute_journal_centroids(df, tfidf)
    if centroids_df.shape[0] == 0:
        raise RuntimeError("Nie udało się obliczyć centroidów dla żadnego czasopisma (brak danych).")

    input_vec = prepare_input_vector(title, abstract, tfidf)

    centroid_matrix = np.vstack(centroids_df['centroid'].values)
    input_dense = np.asarray(input_vec.todense()).reshape(1, -1)
    sims = cosine_similarity(input_dense, centroid_matrix).ravel()

    centroids_df = centroids_df.copy()
    centroids_df['similarity'] = sims

    agg_if = df.groupby('journal')['impact_factor'].agg(['mean', 'count']).rename(columns={'mean': 'if', 'count': 'n_articles_total'})
    results = centroids_df.merge(agg_if, left_index=True, right_index=True, how='left')

    results = results.sort_values(by='similarity', ascending=False).reset_index()
    out = results[['journal', 'n_articles', 'if', 'similarity']].head(top_n)
    out['similarity'] = out['similarity'].round(4)
    if 'if' in out.columns:
        out['if'] = out['if'].round(3)
    return out

# Sekcja uruchomieniowa: przykład, zapis CSV i wykres
if __name__ == "__main__":
    sample_title = "Improve stochastic modeling for external atomic clock with hydrogen maser in PPP multi-GNSS solutions"
    #sample_title = "Improve PPP multi-GNSS solutions"
    sample_abstract = None  # możesz podać abstrakt jako string
    top_n = 10

    print("Weryfikacja tytułu:")
    print(sample_title)
    ranking = verify_article(sample_title, sample_abstract, top_n=top_n)

    print("\nTop dopasowań (journal, n_articles_in_centroid, if, similarity):")
    print(ranking.to_string(index=False))

    # Zapis do CSV
    output_csv_path = os.path.join(DATA_DIR, "verification_ranking.csv")
    ranking.to_csv(output_csv_path, index=False)
    print(f"\n[INFO] Ranking zapisano do pliku: {output_csv_path}")

    # Wykres słupkowy podobieństwa (similarity) dla top N
    plt.figure(figsize=(12, 6))
    sns.barplot(x='similarity', y='journal', data=ranking, color='tab:blue', dodge=False)
    plt.title("Top Journal Matches Similarity")
    plt.suptitle(f"Proposed title: {sample_title}", fontsize=10, y=0.98)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Journal")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_plot_path = os.path.join(DATA_DIR, "verification_ranking_plot.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"[INFO] Wykres rankingowy zapisano do pliku: {output_plot_path}")

# # --- Sekcja uruchomieniowa ---
# if __name__ == "__main__":
#     # Lista tytułów do sprawdzenia
#     sample_titles = [
#         "Characteristics of the IGS receiver clock performance from multi-GNSS PPP solutions",
#         "Multi-GNSS PPP solutions with different handling of system-specific receiver clock parameters and inter-system biases",
#         "Stochastic modeling of the receiver clock parameter in Galileo-only and multi-GNSS PPP solutions",
#         "Multi-GNSS time transfer with different modeling of station coordinates and troposphere delays",
#         "Improving the stability of real-time PPP solutions by receiver clock modeling"
#     ]
#
#     top_n = 10
#     fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 25), sharex=True)
#     all_rankings = []
#
#     for i, title in enumerate(sample_titles):
#         print(f"\n[PRZETWARZANIE {i + 1}/5] Tytuł: {title}")
#
#         # POPRAWKA: zmieniono 'sample_abstract' na 'abstract'
#         ranking = verify_article(title, abstract=None, top_n=top_n)
#         print(ranking.to_string(index=False))
#
#         # Dodajemy informację o tytule do rankingu (do pliku CSV)
#         ranking_to_save = ranking.copy()
#         ranking_to_save['input_title'] = title
#         all_rankings.append(ranking_to_save)
#
#         # Rysowanie na odpowiednim subplocie
#         sns.barplot(
#             x='similarity', y='journal', data=ranking,
#             color='tab:blue', dodge=False, ax=axes[i]
#         )
#
#         # Formatowanie pojedynczego wykresu
#         axes[i].set_title(f"Match {i + 1}: {title[:100]}...", fontsize=10, fontweight='bold')
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel("Journal")
#         axes[i].tick_params(axis='y', labelsize=9)
#
#     axes[-1].set_xlabel("Cosine Similarity", fontsize=12)
#
#     plt.suptitle("Journal Recommendation Ranking for Multiple Titles", fontsize=16, y=0.99)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.subplots_adjust(hspace=0.4)  # Odstęp między wykresami
#
#     output_plot_path = os.path.join(DATA_DIR, "verification_ranking_multiple_titles.png")
#     plt.savefig(output_plot_path, dpi=300)  # Wyższa jakość do pracy
#     plt.close()
#     print(f"\n[INFO] Zbiorczy wykres zapisano do: {output_plot_path}")
#
#     final_csv_df = pd.concat(all_rankings)
#     output_csv_path = os.path.join(DATA_DIR, "verification_ranking_multiple.csv")
#     final_csv_df.to_csv(output_csv_path, index=False)
#     print(f"[INFO] Zbiorczy ranking CSV zapisano do: {output_csv_path}")