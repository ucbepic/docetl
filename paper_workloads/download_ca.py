import sqlite3
import requests
import os
from datetime import datetime, timedelta
import modal
import itertools
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Modal app
image = modal.Image.debian_slim().pip_install("requests")
app = modal.App(name="chronicling-america-downloader", image=image)

# Volume for persistent storage
volume = modal.Volume.from_name("chronicling-america-vol", create_if_missing=True)

# SQLite database file path
db_file_path = "/my_vol/chronicling_america.db"


# Function to initialize the database
@app.function(volumes={"/my_vol": volume})
def init_db():
    volume.reload()
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS issues")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            ocr_file TEXT,
            pdf_file TEXT,
            ocr_text TEXT,
            status TEXT
        )
    """
    )
    conn.commit()
    conn.close()
    volume.commit()
    logger.info("Database initialized.")


def download_file_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logger.info(f"Downloaded: {url}")
            return response.content
        else:
            logger.error(
                f"Failed to download: {url} with status code {response.status_code}"
            )
            return None
    except Exception as e:
        logger.error(f"Exception while downloading {url}: {e}")
        return None


@app.function(volumes={"/my_vol": volume})
def fetch_and_download(sublink):
    def generate_unique_filename(url, suffix):
        unique_id = hashlib.md5(url.encode()).hexdigest()
        return f"{unique_id}_{suffix}"

    ocr_url = os.path.join(sublink, "ocr.txt")
    pdf_url = sublink[:-1] + ".pdf"

    ocr_filename = generate_unique_filename(ocr_url, "ocr.txt")
    pdf_filename = generate_unique_filename(pdf_url, "pdf.pdf")

    ocr_content = download_file_content(ocr_url)
    pdf_content = download_file_content(pdf_url)

    ocr_text = ocr_content.decode("utf-8") if ocr_content else None
    status = "success" if ocr_content and pdf_content else "failed"

    return (
        sublink,
        ocr_filename,
        pdf_filename,
        ocr_text,
        ocr_content,
        pdf_content,
        status,
    )


def generate_urls(start_date_str, end_date_str, base_url):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_date = start_date
    urls = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = base_url.format(date_str)
        urls.append(url)
        current_date += timedelta(days=1)

    return urls


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@app.function(volumes={"/my_vol": volume})
def log_results_to_db(results):
    volume.reload()
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Prepare the bulk insert query
    insert_query = """
        INSERT INTO issues (url, ocr_file, pdf_file, ocr_text, status) VALUES (?, ?, ?, ?, ?)
    """

    # Insert all results in bulk
    cursor.executemany(insert_query, results)
    logger.info(f"Logged {len(results)} records to the database.")

    conn.commit()

    # Print number of rows in the database
    cursor.execute("SELECT COUNT(*) FROM issues")
    logger.info(f"Total rows: {cursor.fetchone()[0]}")

    conn.close()

    # Commit volume
    volume.commit()


@app.function(volumes={"/my_vol": volume}, timeout=10000)
def run_downloader():
    start_date = "1872-10-09"
    end_date = "1881-12-31"
    base_url = "https://chroniclingamerica.loc.gov/lccn/sn84031492/{}/ed-1/seq-1/"

    init_db.remote()  # Initialize the database

    page_links = generate_urls(start_date, end_date, base_url)
    logger.info(f"Generated {len(page_links)} page links")

    results = []
    for page_links_batch in batch(page_links, 900):
        for result in fetch_and_download.map(page_links_batch):
            results.append(result)  # Collect results

    volume.reload()

    os.makedirs("/my_vol/ocr_files", exist_ok=True)
    os.makedirs("/my_vol/pdf_files", exist_ok=True)

    # Write all files at once
    for result in results:
        (
            sublink,
            ocr_filename,
            pdf_filename,
            ocr_text,
            ocr_content,
            pdf_content,
            status,
        ) = result
        if ocr_content:
            ocr_file_path = os.path.join("/my_vol/ocr_files", ocr_filename)
            with open(ocr_file_path, "wb") as ocr_file:
                ocr_file.write(ocr_content)
        if pdf_content:
            pdf_file_path = os.path.join("/my_vol/pdf_files", pdf_filename)
            with open(pdf_file_path, "wb") as pdf_file:
                pdf_file.write(pdf_content)

    volume.commit()

    # Log all results to the database in one go
    log_results_to_db.remote(
        [
            (sublink, ocr_filename, pdf_filename, ocr_text, status)
            for sublink, ocr_filename, pdf_filename, ocr_text, ocr_content, pdf_content, status in results
        ]
    )


@app.local_entrypoint()
def main():
    run_downloader.remote()

    print_error_status.remote()


@app.function(volumes={"/my_vol": volume})
def print_error_status():
    volume.reload()
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Print out all counts
    cursor.execute("SELECT status, COUNT(*) FROM issues GROUP BY status")
    counts = cursor.fetchall()
    for status, count in counts:
        logger.info(f"{status}: {count}")

    # Print number of rows in the database
    cursor.execute("SELECT COUNT(*) FROM issues")
    logger.info(f"Total rows: {cursor.fetchone()[0]}")
    conn.close()
