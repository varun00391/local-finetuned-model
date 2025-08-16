import streamlit as st
import requests
import io
import time

# Title and description for the app
st.set_page_config(
    page_title="Image Description App",
    page_icon="📸"
)

st.title("Image Description App 📸")
st.markdown("Upload an image, and our VLM API will generate a detailed description.")

# API endpoint URL
# NOTE: Replace with your actual deployed API URL if not running locally
API_URL = "http://127.0.0.1:8000/describe"

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_column_width=True
    )

    # Show a spinner while the API request is being processed
    with st.spinner("Analyzing image..."):
        try:
            # Prepare the file for the POST request
            files = {'file': uploaded_file.getvalue()}
            
            # Send the image to the API
            response = requests.post(
                API_URL,
                files=files
            )
            
            # Check for a successful response
            if response.status_code == 200:
                # Parse the JSON response
                description_data = response.json()
                description = description_data.get("description", "No description found.")
                
                # Display the description
                st.subheader("Description:")
                st.info(description)
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                st.error(f"Error message: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Please ensure the API is running at {API_URL}.")
            st.error(f"Error details: {e}")