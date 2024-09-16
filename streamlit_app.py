import streamlit as st
import requests
from PIL import Image
import io
import base64

# FastAPI URL
API_URL = 'http://127.0.0.1:8000/predict/'

# Center the title using HTML and CSS
st.markdown("<h1 style='text-align: center;'>Image Classification App - BDC-USK Hitam Legam</h1>", unsafe_allow_html=True)

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Display how many images were uploaded, centered
    st.markdown(f"<p style='text-align: center;'>Uploaded {len(uploaded_files)} images.</p>", unsafe_allow_html=True)

    # Prepare a list to hold images and their corresponding byte representations
    image_bytes_list = []

    # Display each uploaded image and center it
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

        # Convert the image to bytes and append to list for sending to the API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        image_bytes_list.append(img_bytes)

    # Send images to FastAPI for predictions when the button is clicked
    if st.button("Classify Images"):
        # Prepare files for sending to API
        files = [('files', (uploaded_files[i].name, image_bytes_list[i], 'image/png')) for i in range(len(uploaded_files))]

        # Send the images to FastAPI
        response = requests.post(API_URL, files=files)

        # Check the response and display the predictions
        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            
            # Display predictions with each image
            for i, prediction in enumerate(predictions):
                st.image(Image.open(io.BytesIO(image_bytes_list[i])), caption=f"Prediction: {prediction['class']}", use_column_width=True)
                st.markdown(f"<h2 style='text-align: center; color: green;'>Predicted Class: {prediction['class']}</h2>", unsafe_allow_html=True)
        else:
            st.write(f"Error: {response.status_code}")
