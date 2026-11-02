import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ================= KONFIGURÁCIÓ =================
API_KEY = "AIzaSyCsTuxLPr1tmpZkiaQb00vtsCybVy0clvg"
MAX_KOMMENT = 15000
OUTPUT_DIR = "nyers_adatok"
# =================================================


def get_video_id() -> str:

    """Bekéri a felhasználótól a videó azonosítóját vagy linkjét."""

    user_input = input("Add meg a YouTube videó ID-t vagy linket: ").strip()

    # Explicit ellenőrzés: ha üres, azonnal kivételt dobunk
    if not user_input:
        raise ValueError("A videó azonosítója nem lehet üres!")

    # Ha a felhasználó a teljes linket másolta be:
    if "v=" in user_input:
        video_id = user_input.split("v=")[1].split("&")[0]
    elif "youtu.be/" in user_input:
        video_id = user_input.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = user_input

    # Végső ellenőrzés a kinyert ID-ra
    if not video_id:
        raise ValueError("Nem sikerült kinyerni az azonosítót a megadott linkből!")

    return video_id


def fetch_comments(api_key, video_id, max_komment):
    youtube = build("youtube", "v3", developerKey=api_key)

    kommentek = []
    next_page_token = None
    letoltes_szama = 0

    print(f"\nMaximum {max_komment} komment letöltése a(z) '{video_id}' azonosítójú videóhoz...")

    while letoltes_szama < max_komment:
        try:
            # Mennyit kérjünk a következő hívásban?
            # vagy 100-at, vagy amennyi maradt még a limithez
            requestnbr = min(100, max_komment - letoltes_szama)

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=requestnbr,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()

            items = response.get("items", [])
            for item in items:
                szoveg = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                kommentek.append(szoveg)
                letoltes_szama += 1

            print(f"   {letoltes_szama} / {max_komment} komment letöltve...")

            # Következő oldal
            next_page_token = response.get("nextPageToken")

            # Ha nincs több komment az API-ban, vagy elértük a limitet
            if not next_page_token or letoltes_szama >= max_komment:
                break

        except HttpError as e:
            print(f"Hiba az API hívás közben: {e}")
            break

    return kommentek


if __name__ == "__main__":

    # 1. Videó azonosító bekérése
    video_id = get_video_id()

    # 2. Kimeneti mappa létrehozása (ha még nem létezik) és útvonal összeállítása
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}.csv")

    # 3. Kommentek letöltése
    kommentek = fetch_comments(API_KEY, video_id, MAX_KOMMENT)

    if kommentek:
        print(f"\n✅ Összesen {len(kommentek)} komment letöltve.")

        df = pd.DataFrame(kommentek, columns=["komment_szovege"])
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"📁 Mentve: {output_file}")
    else:
        print("\nNem sikerült kommenteket letölteni (hibás ID?).")