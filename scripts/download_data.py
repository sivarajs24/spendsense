"""Dataset download and sampling script for SpendSense.

Downloads and extracts a manageable sample from the primary Hugging Face
Financial Transaction Categorization dataset (mitulshah/transaction-categorization).

Handles Hugging Face gated dataset authentication (HF_TOKEN) and provides a
zero-auth synthetic generator fallback to ensure users are never blocked.
"""

import argparse
import os
from pathlib import Path
import random
import duckdb
import pandas as pd
import requests

from spendsense.config import get_config

HUGGINGFACE_PARQUET_URL = (
    "https://huggingface.co/datasets/mitulshah/transaction-categorization/resolve/main/data/train-00000-of-00001.parquet"
)


def download_file(url: str, dest_path: Path, hf_token: str | None = None):
    """Downloads a file with streaming to prevent memory exhaustion."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading dataset from: {url}")
    print(f"[*] Destination: {dest_path}")

    headers = {}
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("[*] Using Hugging Face authentication token.")

    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r    Progress: {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)", end="")
    print("\n[+] Download completed successfully!")


def sample_parquet_dataset(parquet_path: Path, output_csv_path: Path, sample_size: int = 150000, random_seed: int = 42):
    """Uses DuckDB to query and sample Parquet without loading millions of rows into RAM."""
    print(f"[*] Sampling {sample_size:,} rows using DuckDB (zero RAM exhaustion)...")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    query = f"""
        SELECT 
            transaction_description AS raw_description,
            category AS true_category,
            country,
            currency
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE transaction_description IS NOT NULL 
          AND category IS NOT NULL
        USING SAMPLE {sample_size} (reservoir, {random_seed});
    """

    sample_df = con.execute(query).df()
    sample_df.to_csv(output_csv_path, index=False)
    print(f"[+] Sample saved to: {output_csv_path} ({len(sample_df):,} rows)")
    print("\nClass distribution in sampled data:")
    print(sample_df["true_category"].value_counts().to_string())


def generate_rich_fallback_dataset(output_csv_path: Path, n_samples: int = 25000, random_seed: int = 42):
    """Generates a rich, zero-auth synthetic dataset across all 10 canonical categories.

    Includes realistic Indian UPI formats, NEFT, POS narrations, and diverse merchants.
    """
    print(f"[*] Generating {n_samples:,} rich multi-category transactions...")
    random.seed(random_seed)

    merchants_by_category = {
        "Food & Dining": [
            "SWIGGY", "ZOMATO", "MCDONALDS", "STARBUCKS", "DOMINOS PIZZA", "KFC",
            "BLINKIT", "ZEPTO", "INSTAMART", "BIGBASKET", "CHAAYOS", "HALDIRAMS",
            "BARBEQUE NATION", "SUBWAY", "PIZZA HUT", "BURGER KING", "CAFE COFFEE DAY"
        ],
        "Transportation": [
            "UBER", "OLA CABS", "RAPIDO BIKE", "IRCTC", "SHELL PETROL", "HPCL FUEL",
            "BPCL OUTLET", "FASTAG TOLL", "INDIGO AIRLINES", "AIR INDIA", "METRO RAIL",
            "AUTO RICKSHAW", "PARKING CHARGES", "VISTARA AIRWAYS"
        ],
        "Shopping & Retail": [
            "AMAZON SELLER", "FLIPKART INTERNET", "MYNTRA DESIGNS", "AJIO TRENDS",
            "MEESHO", "ZARA RETAIL", "H&M CLOTHING", "CROMA ELECTRONICS", "RELIANCE DIGITAL",
            "DECATHLON SPORTS", "IKEA HOME", "NYKAA BEAUTY", "LIFESTYLE STORE"
        ],
        "Entertainment & Recreation": [
            "NETFLIX.COM", "SPOTIFY PREMIUM", "DISNEY HOTSTAR", "AMAZON PRIME",
            "BOOKMYSHOW", "PVR CINEMAS", "INOX LEISURE", "YOUTUBE PREMIUM",
            "STEAM GAMES", "PLAYSTATION SONY", "FITNESS FIRST GYM", "CULT FIT GYM"
        ],
        "Healthcare & Medical": [
            "APOLLO PHARMACY", "MEDPLUS STORE", "TATA 1MG", "PHARMEASY",
            "FORTIS HOSPITAL", "MANIPAL CLINIC", "DR LAL PATHLABS", "NETMEDS",
            "DENTAL CLINIC", "MAX HEALTHCARE", "EYE CARE HOSPITAL"
        ],
        "Utilities & Services": [
            "AIRTEL BROADBAND", "JIO FIBER RECHARGE", "VODAFONE IDEA", "BESCOM ELECTRICITY",
            "TATA POWER", "INDANE GAS", "HP GAS REFILL", "MUNICIPAL WATER",
            "ACT FIBERNET", "URBAN COMPANY", "PIPED NATURAL GAS"
        ],
        "Financial Services": [
            "ZERODHA BROKING", "GROWW INVESTMENTS", "ANGELONE TRADING", "CRED CLUB",
            "HDFC CREDIT CARD", "SBI CARDS", "ICICI PRUDENTIAL LIFE", "LIC OF INDIA",
            "HDFC HOME LOAN", "BAJAJ FINANCE EMI", "MUTUAL FUND SIP"
        ],
        "Income": [
            "INFOSYS SALARY CREDIT", "TCS MONTHLY PAYROLL", "WIPRO SALARY", "UPWORK FREELANCE",
            "SAVINGS INTEREST CREDIT", "DIVIDEND TATA MOTORS", "FLIPKART REFUND",
            "UPI CASHBACK REWARD", "TRAVEL EXPENSE REIMBURSEMENT", "STIPEND CREDIT"
        ],
        "Government & Legal": [
            "INCOME TAX CHALLAN", "GST TAX PAYMENT", "TRAFFIC POLICE CHALLAN",
            "PASSPORT SEVA KENDRA", "MUNICIPAL PROPERTY TAX", "STAMP DUTY REGISTRATION",
            "DISTRICT COURT FEES", "RTO VEHICLE CHARGES", "WATER SEWERAGE TAX"
        ],
        "Charity & Donations": [
            "PM CARES RELIEF", "CRY CHILD RIGHTS", "UNICEF INDIA", "GOONJ FOUNDATION",
            "AKSHAYA PATRA MEALS", "HELPAGE INDIA", "SAVE THE CHILDREN",
            "RAMAKRISHNA MISSION", "SMILE FOUNDATION", "PRATHAM EDUCATION"
        ],
    }

    vpa_handles = ["@ybl", "@okhdfcbank", "@okaxis", "@paytm", "@icici", "@upi", "@oksbi"]
    prefixes = ["UPI/", "UPI-DR-", "UPI/P2M/", "POS ", "ECOM/", "NEFT/", "BILLDESK/"]
    cities = ["BANGALORE", "MUMBAI", "DELHI", "HYDERABAD", "PUNE", "CHENNAI", "KOLKATA", "GURGAON"]

    rows = []
    samples_per_cat = n_samples // len(merchants_by_category)

    for cat, merchants in merchants_by_category.items():
        for _ in range(samples_per_cat):
            merchant = random.choice(merchants)
            style = random.choice(["upi", "pos", "neft", "standard"])

            if style == "upi":
                ref = random.randint(1000000000, 9999999999)
                handle = merchant.lower().replace(" ", "").replace(".", "")[:8]
                vpa = f"{handle}{random.choice(vpa_handles)}"
                desc = f"{random.choice(prefixes)}{ref}/{merchant}/{vpa}"
            elif style == "pos":
                card = f"4012XXXXXXXX{random.randint(1000, 9999)}"
                city = random.choice(cities)
                desc = f"POS {card} {merchant} {city} IN"
            elif style == "neft":
                ref = f"REF{random.randint(100000, 999999)}"
                desc = f"NEFT/HDFC000123/{ref}/{merchant}"
            else:
                desc = f"{merchant} {random.choice(cities)} ONLINE TRANSACTION"

            rows.append({
                "raw_description": desc,
                "true_category": cat,
                "country": "India",
                "currency": "INR",
            })

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[+] Fallback dataset saved to: {output_csv_path} ({len(df):,} rows)")
    print("\nClass distribution:")
    print(df["true_category"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Download or generate the SpendSense training dataset.")
    parser.add_argument("--sample-size", type=int, default=150000, help="Number of rows to sample from Hugging Face.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token for gated dataset.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a rich local synthetic dataset without needing Hugging Face.")
    parser.add_argument("--synthetic-count", type=int, default=25000, help="Number of synthetic rows to generate (default 25,000).")
    args = parser.parse_args()

    config, _ = get_config()
    raw_dir = Path(config.paths.raw_data_dir)
    parquet_path = raw_dir / "hf_transaction_data.parquet"
    sampled_csv_path = raw_dir / f"dataset_a_sample_{args.sample_size}.csv"
    fallback_csv_path = raw_dir / f"dataset_synthetic_{args.synthetic_count}.csv"

    if args.generate_synthetic:
        generate_rich_fallback_dataset(fallback_csv_path, n_samples=args.synthetic_count)
        return

    try:
        if not parquet_path.exists():
            download_file(HUGGINGFACE_PARQUET_URL, parquet_path, hf_token=args.hf_token)

        sample_parquet_dataset(
            parquet_path=parquet_path,
            output_csv_path=sampled_csv_path,
            sample_size=args.sample_size,
            random_seed=config.app.random_seed,
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("\n" + "!" * 70)
            print("[!] Hugging Face returned 401 Unauthorized (Gated Dataset).")
            print("[!] 'mitulshah/transaction-categorization' requires accepting terms on Hugging Face.")
            print("[!]")
            print("[!] You have two options:")
            print("[!]  Option A: Provide your free Hugging Face token:")
            print("[!]      python scripts/download_data.py --hf-token YOUR_TOKEN")
            print("[!]")
            print("[!]  Option B (Recommended - Zero auth, zero wait): Generate 25,000 realistic rows locally:")
            print("[!]      python scripts/download_data.py --generate-synthetic")
            print("!" * 70 + "\n")
        else:
            print(f"[!] HTTP Error: {e}")


if __name__ == "__main__":
    main()
