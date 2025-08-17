import streamlit as st
import requests
import json

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/extract_invoice"

st.set_page_config(page_title="Invoice Parser", page_icon="🧾", layout="centered")

st.title("🧾 Invoice Parser App")
st.write("Paste your invoice details in the text area below in the given format, and get structured JSON output.")

# Default example invoice text
default_invoice = """Invoice ID: 5
Date: 2025-02-01
Vendor: Blue Corp
Address: 123 Main St, NY
Customer: John Smith
Customer Address: 88 Park Ave, NY
Items:
  1. Laptop - 2 pcs - $1200.50
  2. Mouse - 5 pcs - $50.00
Total: $1250.50"""

# Text area for invoice input
invoice_text = st.text_area("Invoice Text:", value=default_invoice, height=250)

if st.button("Extract Invoice Data"):
    if invoice_text.strip():
        with st.spinner("Processing..."):
            try:
                # Send request to FastAPI
                response = requests.post(API_URL, json={"invoice_text": invoice_text})

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Invoice parsed successfully!")

                    # Pretty print JSON result
                    st.subheader("Parsed JSON Output")
                    st.json(result)

                    # Allow user to download result as JSON
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result, indent=4),
                        file_name="invoice_parsed.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")
    else:
        st.warning("Please paste invoice text first.")
