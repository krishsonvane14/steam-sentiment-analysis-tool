import argparse
import json
import os
import re
import pandas as pd
from textblob import TextBlob
from typing import Optional, Dict

REVIEWS_OUT = "data/reviews.jsonl"

TARGET_GAMES_INFO = {
    # Action
    578080: {"genre": "Action", "name": "PUBG: BATTLEGROUNDS"},
    271590: {"genre": "Action", "name": "Grand Theft Auto V"},
    
    # RPG
    292030: {"genre": "RPG", "name": "The Witcher 3: Wild Hunt"},
    377160: {"genre": "RPG", "name": "Fallout 4"},
    435150: {"genre": "RPG", "name": "Divinity: Original Sin 2"},
    489830: {"genre": "RPG", "name": "The Elder Scrolls V: Skyrim"},

    # Simulation
    227300: {"genre": "Simulation", "name": "Euro Truck Simulator 2"},
    413150: {"genre": "Simulation", "name": "Stardew Valley"},
    4000: {"genre": "Simulation", "name": "Garry's Mod"},
    
    # Strategy
    289070: {"genre": "Strategy", "name": "Sid Meier's Civilization VI"},
    1158310: {"genre": "Strategy", "name": "Crusader Kings III"},
    281990: {"genre": "Strategy", "name": "Stellaris"},
    
    # Survival
    252490: {"genre": "Survival", "name": "Rust"},
    105600: {"genre": "Survival", "name": "Terraria"},
}
# -------------------------


def ensure_dirs():
    os.makedirs("data", exist_ok=True)

def clean_text(text: str) -> str:
    """Cleans raw review text for TextBlob."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"httpsS+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_sentiment_label(text: str) -> Optional[str]:
    """Analyzes text polarity using TextBlob."""
    if not text:
        return None
    
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return None  

def main():
    parser = argparse.ArgumentParser(description="Process the large (najzeko) Kaggle Steam reviews CSV.")
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to the large 10GB Kaggle CSV file (e.g., ~/Downloads/steam_reviews.csv)"
    )
    parser.add_argument(
        "--reviews-per-game",
        type=int,
        default=10000,
        help="Number of reviews to save per game."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=REVIEWS_OUT,
        help="Output JSONL file path."
    )
    args = parser.parse_args()

    ensure_dirs()

    target_appids = set(TARGET_GAMES_INFO.keys())
    review_counts = {appid: 0 for appid in target_appids}
    
    total_saved = 0
    games_completed = 0

    print(f"Starting processing for {len(target_appids)} games...")
    print(f"Targeting {args.reviews_per_game} English reviews per game.") 
    print(f"Reading from: {args.csv_file} (This will take a long time)...")

    try:
        with open(args.out, "w", encoding="utf-8") as out_f:
            chunksize = 100_000
            

            with pd.read_csv(args.csv_file, chunksize=chunksize, 
                             usecols=['app_id', 'review', 'language']) as csv_reader:
                
                for chunk in csv_reader:
                    chunk = chunk[chunk['language'] == 'english']
                    if chunk.empty:
                        continue

                    filtered_chunk = chunk[chunk['app_id'].isin(target_appids)]
                    if filtered_chunk.empty:
                        continue
                    
                    for _, row in filtered_chunk.iterrows():
                        if 'app_id' not in row:
                            continue
                        
                        try:
                            appid = int(row['app_id'])
                        except ValueError:
                            continue 
                        
                        if appid not in target_appids:
                            continue

                        if review_counts[appid] >= args.reviews_per_game:
                            continue

                        raw_review_text = row.get("review", "")
                        cleaned_text = clean_text(raw_review_text)
                        sentiment = get_sentiment_label(cleaned_text)

                        if sentiment is None:
                            continue
                        
                        output_record = {
                            "review": raw_review_text,
                            "__appid": appid,
                            "sentiment": sentiment,
                            "genre": TARGET_GAMES_INFO[appid]["genre"]
                        }
                        out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        
                        review_counts[appid] += 1
                        total_saved += 1
                        
                        if review_counts[appid] == args.reviews_per_game:
                            games_completed += 1
                            print(f"  -> Completed appid {appid} ({TARGET_GAMES_INFO[appid]['name']})! ({review_counts[appid]} reviews)")

                    if games_completed == len(target_appids):
                        print("All target games have reached their review limit.")
                        break

    except FileNotFoundError:
        print(f"Error: The reviews file was not found at {args.csv_file}")
        print("Please download it from 'najzeko/steam-reviews-2021' on Kaggle.")
        return
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return
            
    print("\nProcessing Complete.")
    print(f"Total reviews saved: {total_saved}")
    print(f"Final file: {args.out}")
    print("Review counts per game:")
    for appid, count in review_counts.items():
        print(f"  {TARGET_GAMES_INFO[appid]['name']}: {count}")
        
    if games_completed < len(target_appids):
        print("\nNote: Some games may have fewer than 10,000 reviews if the dataset ran out of entries for them.")

if __name__ == "__main__":
    main()