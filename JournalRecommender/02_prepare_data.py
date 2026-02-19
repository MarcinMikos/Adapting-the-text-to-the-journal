"""
Zadaniem skryptu jest uporządkować dane uzyskane z 01_download_information_about_journals.py
"""

# Importowane biblioteki
import os
import glob
import pandas as pd
import numpy as np

# Ścieżki do głównego folderu / miejsca z wyganami plikami / miejsce z dla danych utworzonych
DATA_DIR = "C:\Marcin\Podyplomowka_AnalizaDanych\Project_journalRecommender\JournalRecommender\data"
INPUT_PATTERN = os.path.join(DATA_DIR, "list_of_data_*.csv")
JOURNALS_FILE = os.path.join(DATA_DIR, "journals.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "prepared_data.csv")


def load_all_data(pattern: str) -> pd.DataFrame:
    """Łączy wszystkie pliki CSV z artykułami w jeden DataFrame"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików artykułów: {pattern}")

    print(f"Wczytywanie artykułów z {len(files)} plików...")
    dfs = [pd.read_csv(f, dtype=str) for f in files]
    return pd.concat(dfs, ignore_index=True)


def merge_with_journals(articles_df: pd.DataFrame, journals_path: str) -> pd.DataFrame:
    """Łączy dane artykułów z metadanymi czasopism (punkty, IF)"""
    if not os.path.exists(journals_path):
        print(f"⚠️ Ostrzeżenie: Nie znaleziono pliku {journals_path}. Pomijam łączenie.")
        return articles_df

    print(f"Wczytywanie metadanych czasopism z: {journals_path}")
    jdf = pd.read_csv(journals_path)

    # Standaryzacja nazw kolumn do złączenia
    articles_df['journal'] = articles_df['journal'].astype(str).str.strip().str.lower()
    jdf['journal'] = jdf['journal'].astype(str).str.strip().str.lower()

    # Łączenie (Left Join - zachowujemy wszystkie artykuły)
    combined_df = pd.merge(articles_df, jdf, on='journal', how='left')

    print(f"Połączono dane. Dodano kolumny: {list(jdf.columns[jdf.columns != 'journal'])}")
    return combined_df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Czyszczenie i konwersja na dane numeryczne"""
    print("\n=== Czyszczenie i Inżynieria Cech ===")

    # 1. Usuwanie duplikatów
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Konwersja statusu na 0/1
    if 'status' in df.columns:
        df['is_open_access'] = df['status'].apply(lambda x: 1 if str(x).strip().lower() == 'openaccess' else 0)

    # 3. Konwersja punktów i IF na liczby (obsługa błędów)
    if 'minister_points' in df.columns:
        df['minister_points'] = pd.to_numeric(df['minister_points'], errors='coerce').fillna(0)

    if 'impact_factor' in df.columns:
        # Zamiana przecinków na kropki, jeśli występują
        df['impact_factor'] = df['impact_factor'].astype(str).str.replace(',', '.')
        df['impact_factor'] = pd.to_numeric(df['impact_factor'], errors='coerce').fillna(0.0)

    # 4. Statystyki tekstowe
    df['title_word_count'] = df['title'].fillna("").apply(lambda x: len(str(x).split()))
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(str(x).split()))

    return df


def main():
    print("=" * 60)
    print("ETAP 2: PRZYGOTOWANIE I INTEGRACJA DANYCH")
    print("=" * 60)

    # 1. Załaduj artykuły
    df_articles = load_all_data(INPUT_PATTERN)
    print(f"Pobrano łącznie {len(df_articles)} artykułów.")

    # 2. Połącz z journals.csv
    df_combined = merge_with_journals(df_articles, JOURNALS_FILE)

    # 3. Wyczyść i przygotuj dane
    df_final = clean_and_prepare(df_combined)

    # 4. Zapisz
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nFinalny plik: {OUTPUT_FILE}")
    print(f"Liczba kolumn: {len(df_final.columns)}")
    print(f"Kolumny: {list(df_final.columns)}")


if __name__ == "__main__":
    main()