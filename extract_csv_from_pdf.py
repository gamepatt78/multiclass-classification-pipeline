"""
Utility script to extract iris.csv from the Chrome-exported PDF file:
raw.githubusercontent.com_mwaskom_seaborn-data_master_iris.csv.pdf
"""

import os
import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "raw.githubusercontent.com_mwaskom_seaborn-data_master_iris.csv.pdf")
CSV_PATH = os.path.join(BASE_DIR, "iris.csv")

def extract_iris_csv():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}")
        return False

    print(f"Reading PDF from: {PDF_PATH}")
    doc = pymupdf.open(PDF_PATH)
    full_text = "\n".join([page.get_text() for page in doc])
    
    # Filter for valid CSV rows (5 comma-separated values)
    raw_lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    csv_rows = [line for line in raw_lines if len(line.split(",")) == 5]
    
    print(f"Extracted {len(csv_rows)} CSV lines (1 header + {len(csv_rows) - 1} data rows).")
    
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
        
    print(f"Successfully saved extracted CSV to: {CSV_PATH}")
    return True

if __name__ == "__main__":
    extract_iris_csv()
