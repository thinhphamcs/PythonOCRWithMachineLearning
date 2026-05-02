import sys
from pathlib import Path
import os
import tempfile
import re
import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Initialize Model
MODEL_PATH = os.getenv("MODEL_OUTPUT_PATH")
if not MODEL_PATH:
    st.error("MODEL_OUTPUT_PATH not found in .env file.")
    st.stop()

model = YOLO(MODEL_PATH)

def parse_transaction_line(line):
    """Flexible parser for Transactions and Daily Ledgers."""
    clean_line = line.strip().strip("-| ").strip()
    if not clean_line:
        return None

    amount_match = re.findall(r"(-?[\d,]+\.\d{2})", clean_line)
    amount = amount_match[-1] if amount_match else ""
    
    date_match = re.search(r"^(\d{2}/\d{2}(?:/\d{2,4})?)", clean_line)
    date = date_match.group(1) if date_match else ""
    
    description = clean_line
    if date:
        description = description.replace(date, "", 1)
    if amount:
        description = description.rsplit(amount, 1)[0]
    
    description = description.strip().strip("-| ").strip()

    return {
        "Date": date if date else "N/A",
        "Description": description if description else "---",
        "Amount": f"${amount}" if amount else "N/A"
    }

def process_items(image):
    results = model.predict(image, conf=0.30, verbose=False)
    page_data = []
    
    if len(results) > 0 and results[0].boxes:
        boxes = results[0].boxes
        best_detections = {}
        # Added Blank_Page to tracking
        multi_detections = {"Account_Transactions": [], "Account_Image": [], "Blank_Page": []}

        for box in boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label.lower() == "ignore": continue
            
            if label in ["Account_Header", "Account_Summary"]:
                if label not in best_detections or conf > float(best_detections[label].conf[0]):
                    best_detections[label] = box
            elif label in multi_detections:
                multi_detections[label].append(box)

        # 1. Check for Blank Page First (Requirement 2)
        if multi_detections["Blank_Page"]:
            for box in multi_detections["Blank_Page"]:
                coords = box.xyxy[0].tolist()
                crop = image.crop((coords[0], coords[1], coords[2], coords[3]))
                text = pytesseract.image_to_string(crop).lower()
                if "blank" in text:
                    return [{"label": "is_blank", "type": "status", "content": "This page intentionally left blank"}]

        # 2. Process Metadata (Only if no images exist - Requirement 1)
        if not multi_detections["Account_Image"]:
            for label in ["Account_Header", "Account_Summary"]:
                if label in best_detections:
                    box = best_detections[label]
                    coords = box.xyxy[0].tolist()
                    crop = image.crop((coords[0], coords[1], coords[2], coords[3]))
                    text = pytesseract.image_to_string(crop, lang="eng").strip()
                    if text:
                        page_data.append({"label": label, "type": "text", "content": text})

        # 3. Process Transactions
        if multi_detections["Account_Transactions"]:
            sorted_trans = sorted(multi_detections["Account_Transactions"], key=lambda b: b.xyxy[0][1])
            for box in sorted_trans:
                coords = box.xyxy[0].tolist()
                crop = image.crop((coords[0], coords[1], coords[2], coords[3]))
                text = pytesseract.image_to_string(crop, lang="eng", config='--psm 6').strip()
                if text:
                    page_data.append({"label": "Account_Transactions", "type": "text", "content": text})

        # 4. Process Images
        for box in multi_detections["Account_Image"]:
            coords = box.xyxy[0].tolist()
            crop = image.crop((coords[0], coords[1], coords[2], coords[3]))
            page_data.append({"label": "Account_Image", "type": "image", "content": crop})

    return page_data

def display_results(items):
    # Handle Blank Page status first
    if any(it["label"] == "is_blank" for it in items):
        st.info("This page intentionally left blank")
        return

    # Metadata (Header/Summary)
    metadata = [it for it in items if it["label"] in ["Account_Header", "Account_Summary"]]
    if metadata:
        summary_rows = [{"Field": it["label"].replace("_", " "), "Information": it["content"].replace("\n", " ")} for it in metadata]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Transactions
    transactions = [it for it in items if it["label"] == "Account_Transactions"]
    if transactions:
        parsed_data = []
        for trans_block in transactions:
            lines = [line.strip() for line in trans_block["content"].split("\n") if line.strip()]
            for line in lines:
                res = parse_transaction_line(line)
                if res: parsed_data.append(res)
        
        if parsed_data:
            st.dataframe(pd.DataFrame(parsed_data), use_container_width=True, hide_index=True)

    # Images
    images = [it for it in items if it["type"] == "image"]
    if images:
        for img in images:
            st.image(img["content"], use_column_width=True)

def main():
    st.set_page_config(page_title="OCR Parser", layout="wide")
    st.markdown("<style>[data-testid='stHeader'] {display:none;} .block-container {padding-top: 2rem;} [data-testid='stElementToolbar'] { display: none; }</style>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload", type=["pdf", "png", "jpg", "tiff"], label_visibility="collapsed")

    if uploaded_file:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            with st.spinner("Processing..."):
                images = convert_from_path(tmp_path, dpi=300)
                for i, img in enumerate(images):
                    page_items = process_items(img)
                    if page_items:
                        with st.expander(f"Page {i+1}", expanded=True):
                            display_results(page_items)
            os.unlink(tmp_path)
        else:
            with st.spinner("Processing..."):
                img = Image.open(uploaded_file).convert("RGB")
                page_items = process_items(img)
                display_results(page_items)

if __name__ == "__main__":
    main()