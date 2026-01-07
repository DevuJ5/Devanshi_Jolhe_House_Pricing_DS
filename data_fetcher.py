import os
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



TRAIN_PATH = Path("data/raw/train(1).xlsx")
TEST_PATH  = Path("data/raw/test2.xlsx")

IMG_TRAIN_DIR = Path("data/images/train")
IMG_TEST_DIR  = Path("data/images/test")
IMG_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
IMG_TEST_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("outputs/preds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Choose consistent settings
ZOOM = 18
SIZE = "256x256"
SCALE = 2
MAPTYPE = "satellite"
FORMAT = "png"

def build_url(lat, lon):
    return (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}"
        f"&zoom={ZOOM}"
        f"&size={SIZE}"
        f"&scale={SCALE}"
        f"&maptype={MAPTYPE}"
        f"&format={FORMAT}"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

def make_session():
    s = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def download_image(session, url, out_path, timeout=20):
    r = session.get(url, timeout=timeout)
    if r.status_code == 200 and len(r.content) > 5000:
        out_path.write_bytes(r.content)
        return True, None
    return False, f"HTTP {r.status_code}, bytes={len(r.content)}"


def fetch(df, out_dir, id_col="id", lat_col="lat", lon_col="long",
          max_workers=10, chunk_size=1000):
    failures = []
    session = make_session()

    # Build a list of rows that actually need downloading (skip existing files)
    todo = []
    for _, row in df.iterrows():
        pid = row[id_col]
        out_path = out_dir / f"{pid}.png"
        if out_path.exists():
            continue
        todo.append(row)

    if not todo:
        return failures

    def process_row(row):
        pid = row[id_col]
        lat = row[lat_col]
        lon = row[lon_col]

        if pd.isna(lat) or pd.isna(lon):
            return (pid, "missing lat/lon")

        out_path = out_dir / f"{pid}.png"
        url = build_url(lat, lon)
        ok, err = download_image(session, url, out_path)

        # If rate-limited, small sleep helps
        if (not ok) and ("HTTP 429" in str(err)):
            time.sleep(0.2)
        return None if ok else (pid, err)

    # Chunking so you don't submit 10k+ futures at once
    for start in range(0, len(todo), chunk_size):
        batch = todo[start:start+chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_row, row) for row in batch]
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res is not None:
                    failures.append(res)

    return failures


def main():
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY not set. Use: set GOOGLE_MAPS_API_KEY=YOUR_KEY")

    train_df = pd.read_excel(TRAIN_PATH)
    test_df  = pd.read_excel(TEST_PATH)

    # Ensure columns exist
    for col in ["id", "lat", "long"]:
        assert col in train_df.columns, f"Missing {col} in train"
        assert col in test_df.columns, f"Missing {col} in test"

    print("Downloading train images...")
    f_train = fetch(train_df, IMG_TRAIN_DIR, max_workers=10, chunk_size=1000)
    print("Downloading test images...")
    f_test = fetch(test_df, IMG_TEST_DIR, max_workers=10, chunk_size=1000)



    # Save failure logs
    if f_train:
        pd.DataFrame(f_train, columns=["id", "error"]).to_csv(OUT_DIR / "train_image_failures.csv", index=False)
    if f_test:
        pd.DataFrame(f_test, columns=["id", "error"]).to_csv(OUT_DIR / "test_image_failures.csv", index=False)

    print("Done. Failures:", len(f_train) + len(f_test))

if __name__ == "__main__":
    main()
