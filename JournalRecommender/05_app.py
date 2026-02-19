# 05_app.py
"""
Aplikacja Flask udostępniająca prosty interfejs do funkcji weryfikujące dopasowanie tekstu do manuskryptu
z pliku 04_verification_model.py oraz podstawowe statystyki zbioru prepared_data.csv.
Dodatkowo generuje wykres słupkowy podobieństwa dla wyniku weryfikacji i udostępnia go
jako plik w folderze 'static'.
"""

# Importowane biblioteki
import os
import importlib.util
import traceback
from flask import Flask, render_template, request, url_for

import pandas as pd  # do liczenia statystyk
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżki do głównego folderu / miejsca z wyganami plikami /
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(BASE_DIR, "04_verification_model.py")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

# Dynamiczny import modułu 04_verification_model.py pod aliasem "verification_model"
if not os.path.exists(MODULE_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku modułu: {MODULE_PATH}")

spec = importlib.util.spec_from_file_location("verification_model", MODULE_PATH)
verification_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification_model)

# Funkcje z zaimportowanego modułu
verify_article = getattr(verification_model, "verify_article", None)
load_prepared_data = getattr(verification_model, "load_prepared_data", None)

if verify_article is None:
    raise RuntimeError("Moduł verification_model nie zawiera funkcji 'verify_article'.")

# Flask app (szablony w ./templates)
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=STATIC_DIR)

sns.set(style="whitegrid")

def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Oblicza i zwraca podstawowe statystyki z przygotowanego df."""
    stats = {}
    if df is None or df.shape[0] == 0:
        return stats

    df = df.copy()
    stats['total_articles'] = int(len(df))
    stats['unique_journals'] = int(df['journal'].astype(str).str.strip().nunique())

    # is_open_access -> interpretacja różnych formatów
    if 'is_open_access' in df.columns:
        try:
            oa_numeric = pd.to_numeric(df['is_open_access'], errors='coerce').fillna(0).astype(int)
        except Exception:
            oa_numeric = df['is_open_access'].astype(str).str.lower().isin(['1', 'true', 'yes', 't', 'y']).astype(int)
        stats['open_access_count'] = int(oa_numeric.sum())
        stats['open_access_ratio'] = round(float(stats['open_access_count']) / stats['total_articles'], 3) if stats['total_articles'] > 0 else 0.0
    else:
        stats['open_access_count'] = 0
        stats['open_access_ratio'] = 0.0

    # Średnia długość tytułu / abstraktu (jeśli dostępne)
    if 'title_word_count' in df.columns:
        try:
            avg_title = pd.to_numeric(df['title_word_count'], errors='coerce').dropna()
            stats['avg_title_word_count'] = round(float(avg_title.mean()), 2) if len(avg_title) > 0 else None
        except Exception:
            stats['avg_title_word_count'] = None
    else:
        stats['avg_title_word_count'] = None

    if 'abstract_word_count' in df.columns:
        try:
            avg_abs = pd.to_numeric(df['abstract_word_count'], errors='coerce').dropna()
            stats['avg_abstract_word_count'] = round(float(avg_abs.mean()), 2) if len(avg_abs) > 0 else None
        except Exception:
            stats['avg_abstract_word_count'] = None
    else:
        stats['avg_abstract_word_count'] = None

    # Top 10 czasopism wg liczby artykułów
    try:
        top_j = df['journal'].value_counts().head(10)
        stats['top_journals'] = [{'journal': j, 'count': int(c)} for j, c in top_j.items()]
    except Exception:
        stats['top_journals'] = []

    # Publikacje wg roku (jeśli publication_date dostępne)
    if 'publication_date' in df.columns:
        try:
            years = pd.to_datetime(df['publication_date'], errors='coerce').dt.year
            top_years = years.value_counts().sort_index(ascending=False).head(10)
            stats['top_years'] = {int(k): int(v) for k, v in top_years.items() if not pd.isna(k)}
        except Exception:
            stats['top_years'] = {}
    else:
        stats['top_years'] = {}

    return stats


def save_ranking_plot(ranking_df: pd.DataFrame, title_text: str, filename: str = "verification_ranking_plot.png"):
    """
    Generuje i zapisuje wykres słupkowy (similarity vs journal) do folderu static.
    Nadpisuje plik o podanej nazwie.
    """
    if ranking_df is None or ranking_df.shape[0] == 0:
        return None

    plot_path = os.path.join(STATIC_DIR, filename)
    try:
        df_plot = ranking_df.copy()
        plt.figure(figsize=(10, max(4, 0.4 * len(df_plot))))
        ax = plt.gca()
        sns.barplot(x='similarity', y='journal', data=df_plot, color='tab:blue', dodge=False, ax=ax)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("")
        ax.set_xlim(0, 1.0)
        ax.set_title("Top Journal Matches Similarity")
        suptitle = (title_text[:200] + '...') if len(title_text) > 200 else title_text
        plt.suptitle(f"Proposed title: {suptitle}", fontsize=9, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return filename
    except Exception:
        plt.close()
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    title = ""
    ranking = None
    ranking_error = None
    plot_filename = None
    stats = {}
    try:
        if load_prepared_data is not None:
            df_prepared = load_prepared_data()  # korzysta z domyślnej ścieżki w module
            stats = compute_basic_stats(df_prepared)
        else:
            stats = {}
    except Exception as e:
        stats = {'error_loading_data': str(e)}

    # Obsługa formularza POST
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        if title:
            try:
                df_rank = verify_article(title, abstract=None, top_n=10)
                # konwertujemy DataFrame na listę słowników dla Jinja
                if isinstance(df_rank, pd.DataFrame):
                    ranking = df_rank.to_dict(orient="records")
                    # Zapisz również wykres jako plik w static i ustaw nazwę do przekazania do szablonu
                    plot_file = save_ranking_plot(df_rank, title, filename="verification_ranking_plot.png")
                    if plot_file:
                        plot_filename = plot_file
                else:
                    ranking = df_rank
            except Exception as e:
                ranking = None
                ranking_error = str(e) + "\n" + traceback.format_exc()

    return render_template(
        "index.html",
        title=title,
        ranking=ranking,
        ranking_error=ranking_error,
        stats=stats,
        plot_filename=plot_filename
    )


if __name__ == "__main__":
    # Uruchamiaj lokalnie: python 05_app.py
    app.run(debug=True)