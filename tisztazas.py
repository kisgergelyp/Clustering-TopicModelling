import os
import pandas as pd
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ================= KONFIGURÁCIÓ =================
INPUT_DIR = "nyers_adatok"  # feldolgozatlan CSV-k
OUTPUT_DIR = "tisztitott_adatok"  # Ide fognak kerülni a tisztázott kommentek
MIN_SZOHOSSZ = 3  # 3 karakternél rövidebb kommenteketet törli a program


# =================================================


def adatok_betoltese():
    """Bekéri a fájlnevet, és megpróbálja betölteni az INPUT_DIR mappából."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    fajlnev = input(f"Kérlek, add meg a CSV fájl nevét (a '{INPUT_DIR}' mappából): ").strip()

    # Ha a kiterjesztés nélkül van megadva az id, .csv-t hozzáteszi a program
    if not fajlnev.lower().endswith('.csv'):
        fajlnev += '.csv'

    utvonal = os.path.join(INPUT_DIR, fajlnev)

    try:
        df = pd.read_csv(utvonal)
        return df, fajlnev
    except FileNotFoundError:
        print(f"Hiba: Nem található a(z) '{utvonal}' fájl!")
        return None, None


def main():
    # Betöltés
    df, input_file = adatok_betoltese()
    if df is None:
        return

    # Kimeneti mappa létrehozása (ha még nem létezik) és fájlnév generálása
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    alapnev, kiterjesztes = os.path.splitext(input_file)
    output_file = os.path.join(OUTPUT_DIR, f"{alapnev}_tisztazva{kiterjesztes}")

    print(f"Eredeti kommentek száma: {len(df)}")

    #  1. DUPLIKÁTUMOK ÉS NYILVÁNVALÓ SPAM ELTÁVOLÍTÁSA
    df = df.dropna(subset=['komment_szovege']) # Üres sorokat törli (NaN)
    df = df.drop_duplicates(subset=['komment_szovege']) #duplikált sorokat törli (bot spam)

    # felesleges whitespace karakterek eltávolítása, de csak az elejéről és a végéről törli
    df['komment_szovege'] = df['komment_szovege'].astype(str).str.strip()
    df = df[df['komment_szovege'] != ''] # Üres sorokat törli, amiben csak szóköz van

    print(f"Duplikátumok és üresek eltávolítva: {len(df)} db maradt.")

    #  2. ALAP SZÖVEGTISZTÍTÁS (REGEX)
    def clean_basic(text):
        # Biztonsági ellenőrzés, ha bármilyen nem-string maradna
        if not isinstance(text, str):
            return ""

        # URL-ek eltávolítása
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        #  E-mail címek és @-es említések eltávolítása
        text = re.sub(r'@\w+', '', text)

        # felesleges whitespace-ek eltávolítása (szövegben bárhol)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # elvégezzük a szövegtisztízást
    df['clean1'] = df['komment_szovege'].apply(clean_basic)

    # 3. emoji-k eltávolítása (lehetne őket szöveggé alakítani, de téma csoportosításhoz ez felesleges.)
    df['clean2'] = df['clean1'].apply(emoji.replace_emoji, replace='')

    #  4. KISBETŰSÍTÉS ÉS KÜLÖNLEGES KARAKTEREK SZŰRÉSE
    # Csak betűket és szóközöket hagyunk meg (a számok és az írásjelek nem kellenek)
    def keep_letters_only(text):
        # Csak angol betűket (a-z) és szóközt hagyunk meg
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # magyar szöveghez:
        #text = re.sub(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ\s]', '', text)
        return text.lower().strip()

    df['clean3'] = df['clean2'].apply(keep_letters_only)

    #  5. RÖVID ÉS ÉRTELMETLEN KOMMENTEK SZŰRÉSE
    df['szohossz'] = df['clean3'].str.split().str.len()
    df = df[df['szohossz'] >= MIN_SZOHOSSZ]

    print(f"Rövid kommentek kiszűrve: {len(df)} db maradt.")

    # 6. STOPWORDS
    stop_words_eng = set(stopwords.words('english'))
    #stop_words_hun = set(stopwords.words('hungarian'))
    stop_words = stop_words_eng #| stop_words_hun
    # Kiegészítő YouTube-specifikus stopword lista angol, (magyar kikommentelve)
    extra_stops = {'im', 'ive', 'idk', 'dont', 'cant', 'wont', 'didnt', 'youre',
                   'shes', 'hes', 'its', 'that', 'this', 'just', 'like', 'really',
                   'get', 'got', 'know', 'think', 'see', 'want', 'go', 'come', 'make'
    
                    #'amúgy', 'am', 'sztem', 'szerintem', 'tök', 'nagyon', 'csak',
                    #'ugye', 'szóval', 'mondjuk', 'tehát', 'igazából', 'akkor', 'már',
                    #'még', 'is', 'meg', 'egy', 'ez', 'az', 'hogy', 'ha', 'vagy',
                    #'mint', 'olyan', 'ilyen', 'nincs', 'van', 'volt', 'lesz',
                    #'igen', 'nem', 'nekem', 'neked', 'tudom', 'kell', 'jó'
        }

    stop_words.update(extra_stops)

    # 7. LEMMATIZÁLÁS (magyarra nem tudtam megcsinálni)
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        tokens = word_tokenize(text)
        # Szűrés: csak betűkből álló szavak, stopword-ök kihagyása
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        # Lemmatizálás (visszavezetés alapszóra: running -> run, mice -> mouse)
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    df['clean_final'] = df['clean3'].apply(lemmatize_text)

    # 8. VÉGSŐ SZŰRÉS (Ha a lemmatizálás után üres lett a komment)
    df = df[df['clean_final'].str.strip() != '']
    df = df[df['clean_final'].str.split().str.len() >= MIN_SZOHOSSZ]

    print(f"Végső tisztított kommentek száma: {len(df)}")

    #  8. MENTÉS
    # A tisztázott szöveget és a tisztázott + szűrt + lemattizált szöveget külön mentjük
    df[['clean2', 'clean_final']].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Mentve: {output_file}")


if __name__ == "__main__":
    main()