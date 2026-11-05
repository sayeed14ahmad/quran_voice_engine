import requests
import json

print("Downloading the full Quran...")
url = "https://api.alquran.cloud/v1/quran/quran-uthmani"

try:
    response = requests.get(url, timeout=15)
    data = response.json()
    
    quran_db = []
    
    # Loop through all 114 Surahs and 6,236 Ayahs
    for surah in data["data"]["surahs"]:
        for ayah in surah["ayahs"]:
            quran_db.append({
                "surah": surah["number"],
                "ayah": ayah["numberInSurah"],
                "arabic": ayah["text"]
            })
            
    # Save it in the exact format app_2.py expects
    with open("quran_db.json", "w", encoding="utf-8") as f:
        json.dump(quran_db, f, ensure_ascii=False, indent=2)
        
    print(f"Success! Saved {len(quran_db)} verses to quran_db.json")
    
except Exception as e:
    print(f"Failed to download: {e}")