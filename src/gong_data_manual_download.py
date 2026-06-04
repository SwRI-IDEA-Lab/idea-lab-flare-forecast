import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Base URL of GONG magnetogram data
BASE_URL = "https://gong2.nso.edu/ftp/oQR/zqa/"

# Set your desired date range (Adjust as needed for your use case)
START_DATE = "2021-06-16"
END_DATE = "2026-04-20"

def generate_date_list(start_date, end_date):
    """Generate a list of dates between the start and end date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    date_list = []
    while start <= end:
        date_list.append(start.strftime("%Y%m%d"))  # Format YYYYMMDD
        start += delta
    return date_list

def get_gong_files_for_date(base_url, date):
    """Scrape available GONG files for a specific date."""
    date_yyyymm = date[:6]
    date_yymm = date[2:6]
    date_yymmdd = date[2:]
    date_dd = date[-2:]
    subdir = 'bbzqa'+ date_yymmdd
    
    
    url = f'{base_url}{date_yyyymm}/{subdir}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error accessing {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    files = []
    url_dir_list = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".fits.gz"):
        # if href and href.startswith(f"bbbzi{date_yymmdd}"):  # Magnetogram files start with zqsYYYYMMDD
            files.append(str(date_yyyymm)+'/'+str(subdir)+'/'+href)
    return files

def download_file(url, save_dir):
    """Download a given file and save it to the local filesystem."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = os.path.join(save_dir, url.split("/")[-1])
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Failed to download {url} (HTTP {response.status_code})")
        return None

if __name__ == "__main__":
    # Directory to save downloaded files
    SAVE_DIR = "/d0/jkobayashi/data/gong/manual/magnetograms/zqa"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Generate list of dates to fetch
    daily_date_list = generate_date_list(START_DATE, END_DATE)

    # Loop over dates and download one magnetogram file per day
    for full_date in daily_date_list:
        print(f"Processing date: {full_date}")

        # Get available magnetogram files for the date
        files = get_gong_files_for_date(BASE_URL, full_date)
        if not files:
            print(f"No files found for date {full_date}")
            continue

        # Download the first magnetogram file for the day
        file_url = BASE_URL + files[0]  # Use the first available file
        download_file(file_url, SAVE_DIR)