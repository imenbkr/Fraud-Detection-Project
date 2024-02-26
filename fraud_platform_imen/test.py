import streamlit as st
import base64
import cv2
import numpy as np
import easyocr
from camera_input_live import camera_input_live
from streamlit_card import card
import io

from PIL import Image
import plotly.figure_factory as ff
import json
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import base64

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import random

from spacy import displacy
from fpdf import FPDF
from sklearn import tree
import tempfile
import pytesseract
import os 
# Create a spaCy NLP pipeline
nlp = spacy.load("model-best")

import ner_app #importing the ner_app script
#-----------------------------------------PAGE CONFIGURATIONS--------------------------------------
st.set_page_config(
        page_title="Main page",
)

# Define CSS styles for the sidebar
sidebar_styles = """
    .sidebar-content {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sidebar-image {
        max-width: 150px;
        display: block;
        margin: 0 auto;
    }
"""

def set_background_color(hex_color, color):
    style = f"""
        <style>
        .background-text {{
            background-color: {hex_color};
            padding: 5px; /* Adjust padding as needed */
            border-radius: 5px; /* Rounded corners */
            color: {color}; /* Text color */
        }}
        </style>
    """
    return style
#-----------------------------------------WELCOME PAGE--------------------------------------

st.title("Test App")

st.write("Welcome to Test App")


with st.sidebar:
    st.sidebar.markdown(f"<style>{sidebar_styles}</style>", unsafe_allow_html=True)

    st.write(
        "<div style='display: flex; justify-content: center;'>"
        "<img src='https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png' style='width: 150px;'>"
        "</div>",
        unsafe_allow_html=True
    )

    st.title("PixOCR")
    choice = st.radio("Navigation", ["Extract text from images", "Display labeled text", "Download PDF Summary", "Fraud Detection"], index=0)
    st.info("This project application helps you annotate your medicine data and detect fraud.")
    st.sidebar.success("Select an option above.")
    
#-----------------------------------------RADIO BUTTON CHOICES--------------------------------------

# Define containers for each choice's content
if choice == "Extract text from images":

        choice = st.radio('Choose an option', ["Capture Image with Camera", "Upload Image"])
        if choice == "Capture Image with Camera":
            st.write("# See a new image every second")
            controls = st.checkbox("Show controls")
            image = camera_input_live(show_controls=controls)
            if image is not None:
                #st.write(type(image))
                st.image(image)
                if st.button("Extract Text From Image"):
                        #convert the file to an opencv image.
                        pil_image = Image.open(image)
                        numpy_array = np.array(pil_image)
                        #opencv_image = cv2.imdecode(numpy_array, 1)
                        opencv_image = numpy_array.copy()
                        #st.write(type(opencv_image))

                        # extract text and process the result
                        text, result = ner_app.ocr_extraction(opencv_image)
                        st.session_state = text
                        # display extracted text
                        st.markdown("Here's the Extracted text:")
                        st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                        styled_text = f"<div class='background-text'>{text}</div>"
                        st.markdown(styled_text, unsafe_allow_html=True)

                        # draw contours on the image
                        img = ner_app.draw_contours(opencv_image, result)

                        # display image with contours
                        st.markdown("Here's the image with contours on the text detected:")
                        st.image(img, channels="BGR")
                        
