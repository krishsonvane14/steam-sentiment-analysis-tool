import argparse
import json
import os
import sys
import time
import urllib.parse
from typing import Dict, Iterator, List, Optional


import requests



APPLIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
APPREVIEWS_URL_TMPL = "https://store.steampowered.com/appreviews/{appid}"
APPLIST_OUT = "data/app_list.json"
REVIEWS_OUT = "data/reviews.jsonl"
CHECKPOINT_FILE = "data/checkpoint.json"



def ensure_dirs():
    os.makedirs("data", exist_ok=True)



def get_app_list(session: requests.Session) -> List[Dict]:
    """
    Returns a list of {'appid': int, 'name': str} for all public applications.
    """
    resp = session.get(APPLIST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    apps = data.get("applist", {}).get("apps", [])
    return apps



def iter_app_reviews(
    session: requests.Session,
    appid: int,
    *,
    language: str = "english",
    filter_mode: str = "all",  # 'recent', 'updated', or 'all'
    review_type: str = "all",  # 'all', 'positive', 'negative'
    purchase_type: str = "all",  # 'all', 'steam', 'non_steam_purchase'
    include_offtopic: bool = True,
    num_per_page: int = 100,
    day_range: Optional[int] = 9223372036854775807,
    delay_sec: float = 1.0,
    max_reviews: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Yields raw review dicts for a given appid by paginating with the cursor.
    """
    cursor = "*"
    seen_cursors = set()
    count = 0


    while True:
        params = {
            "json": 1,
            "language": language,
            "filter": filter_mode,
            "review_type": review_type,
            "purchase_type": purchase_type,
            "num_per_page": num_per_page,
        }


        if day_range is not None:
            params["day_range"] = day_range


        # Steam defaults to filtering off-topic activity; toggle to include or exclude
        # 0 = include off-topic activity, 1 = filter it out (naming is counterintuitive)
        params["filter_offtopic_activity"] = 0 if include_offtopic else 1


        # Cursor must be URL-encoded except for the initial '*'
        if cursor != "*":
            enc_cursor = urllib.parse.quote(cursor, safe="")
        else:
            enc_cursor = cursor


        params["cursor"] = enc_cursor
        url = f"{APPREVIEWS_URL_TMPL.format(appid=appid)}"


        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            # Backoff and retry a few times
            for wait in (2, 5, 10):
                try:
                    time.sleep(wait)
                    r = session.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    payload = r.json()
                    break
                except Exception:
                    payload = None
            if payload is None:
                # Give up on this appid page
                return


        reviews = payload.get("reviews", [])
        next_cursor = payload.get("cursor")


        if not reviews:
            return


        for rev in reviews:
            rev["__appid"] = appid  # annotate to keep app context
            yield rev
            count += 1
            if max_reviews is not None and count >= max_reviews:
                return


        if not next_cursor or next_cursor in seen_cursors:
            return


        seen_cursors.add(next_cursor)
        cursor = next_cursor
        time.sleep(delay_sec)



def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_app_index": -1}



def save_checkpoint(state: Dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)



def main():
    parser = argparse.ArgumentParser(description="Collect Steam reviews for many apps into JSONL.")
    parser.add_argument("--max-apps", type=int, default=500, help="Max number of apps to process this run.")
    parser.add_argument("--max-reviews-per-app", type=int, default=500, help="Limit per app to avoid huge pulls.")
    parser.add_argument("--language", type=str, default="english", help="Review language (e.g., english, all).")
    parser.add_argument("--filter", type=str, default="all", choices=["recent", "updated", "all"], help="Review filter.")
    parser.add_argument("--review-type", type=str, default="all", choices=["all", "positive", "negative"], help="Review type.")
    parser.add_argument("--purchase-type", type=str, default="all", choices=["all", "steam", "non_steam_purchase"], help="Purchase type.")
    parser.add_argument("--include-offtopic", action="store_true", help="Include off-topic activity reviews.")
    parser.add_argument("--num-per-page", type=int, default=100, help="Batch size (20 default, 100 max typically).")
    parser.add_argument("--delay-sec", type=float, default=1.0, help="Delay between page requests.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (by app index).")
    parser.add_argument("--out", type=str, default=REVIEWS_OUT, help="Output JSONL file path.")
    parser.add_argument("--applist-out", type=str, default=APPLIST_OUT, help="App list JSON path.")
    parser.add_argument("--skip-no-reviews", action="store_true", default=True, help="Skip apps with 0 reviews (default: True).")
    args = parser.parse_args()


    ensure_dirs()


    session = requests.Session()
    session.headers.update({"User-Agent": "steam-reviews-collector/1.0"})


    # Fetch or load app list
    if os.path.exists(args.applist_out):
        with open(args.applist_out, "r", encoding="utf-8") as f:
            apps = json.load(f)
    else:
        apps = get_app_list(session)
        with open(args.applist_out, "w", encoding="utf-8") as f:
            json.dump(apps, f)


    # Optional resume
    start_index = 0
    if args.resume:
        state = load_checkpoint()
        start_index = max(0, state.get("last_app_index", -1) + 1)


    # Iterate apps and collect reviews
    processed = 0
    skipped = 0
    total_apps = len(apps)
    print(f"Apps available: {total_apps}")
    print(f"Starting from index: {start_index}")


    # Open output in append mode for resumability
    with open(args.out, "a", encoding="utf-8") as out_f:
        for i in range(start_index, min(total_apps, start_index + args.max_apps)):
            app = apps[i]
            appid = app.get("appid")
            name = app.get("name", "")
            if not appid:
                continue


            print(f"[{i}] AppID={appid} Name={name[:60]!r}")


            pulled = 0
            for rev in iter_app_reviews(
                session,
                appid=appid,
                language=args.language,
                filter_mode=args.filter,
                review_type=args.review_type,
                purchase_type=args.purchase_type,
                include_offtopic=args.include_offtopic,
                num_per_page=args.num_per_page,
                day_range=9223372036854775807,
                delay_sec=args.delay_sec,
                max_reviews=args.max_reviews_per_app,
            ):
                out_f.write(json.dumps(rev, ensure_ascii=False) + "\n")
                pulled += 1


            # Skip printing and counting apps with 0 reviews if flag is set
            if args.skip_no_reviews and pulled == 0:
                skipped += 1
                print(f"  -> skipped (0 reviews)")
            else:
                print(f"  -> collected: {pulled} reviews")
                if pulled > 0:
                    processed += 1


            # Save checkpoint after each app
            save_checkpoint({"last_app_index": i})


    print(f"Done. Processed {processed} apps with reviews this run. Skipped {skipped} apps with 0 reviews. Output -> {args.out}")
    print("You can rerun with --resume to continue from the next app.")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
