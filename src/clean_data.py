import pandas as pd
import re
import argparse
import os

DEFAULT_IN_FILE = "data/reviews.jsonl"
DEFAULT_OUT_FILE = "data/reviews_cleaned.csv"

def clean_text(text: str) -> str:
    """
    Cleans raw review text by removing HTML tags, URLs, emojis,
    punctuation, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text) 
    text = re.sub(r"httpsS+|www\S+", " ", text)  
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  
    text = text.lower()  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

def main():
    parser = argparse.ArgumentParser(description="Clean the raw review data.")
    parser.add_argument(
        "--in-file",
        type=str,
        default=DEFAULT_IN_FILE,
        help="Path to the input JSONL file (e.g., data/reviews.jsonl)"
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=DEFAULT_OUT_FILE,
        help="Path for the output cleaned CSV file (e.g., data/reviews_cleaned.csv)"
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    print(f"Loading raw data from {args.in_file}...")
    try:
        df = pd.read_json(args.in_file, lines=True)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.in_file}")
        print("Please run the data extraction script first.")
        return
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return
        
    print(f"Loaded {len(df)} reviews.")

    print("Cleaning review text... This may take a moment.")
    df['cleaned_review'] = df['review'].apply(clean_text)

    df = df[df['cleaned_review'].str.len() > 0].copy()
    
    final_columns = [
        'cleaned_review', 
        'sentiment', 
        'genre', 
        '__appid', 
        'review'  
    ]
    df_cleaned = df[final_columns]

    print(f"Saving cleaned data to {args.out_file}...")
    try:
        df_cleaned.to_csv(args.out_file, index=False)
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return

    print("\nCleaning complete!")
    print(f"Total reviews saved: {len(df_cleaned)}")
    print(f"Output file: {args.out_file}")

if __name__ == "__main__":
    main()