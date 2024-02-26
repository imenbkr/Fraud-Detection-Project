import streamlit as st
import base64
import cv2
import numpy as np
from camera_input_live import camera_input_live

from PIL import Image
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm

import pandas as pd
import random
from spacy import displacy
from fpdf import FPDF

#create a spaCy NLP pipeline
nlp = spacy.load("model-best")

import ner_app #importing the ner_app script
import login_page #importing login_page script
#-----------------------------------------SET BACKGROUND IMAGE--------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def cooling_highlight(val):
    color = '#ACE5EE' if val else '#F2F7FA'
    return f'background-color: {color}'

#-----------------------------------------PAGE CONFIGURATIONS--------------------------------------
#st.set_page_config(
#        page_title="Main page",
#)


#define CSS styles for the sidebar
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

st.title("Medicine Fraud Detection App")
set_background('C:/fraud_platform_imen/widgets/medicine-capsules.png')
st.write("Welcome to Medicine Fraud Detection App")
#-----------------------------------------SIDE BAR--------------------------------------
with st.sidebar:
    st.sidebar.markdown(f"<style>{sidebar_styles}</style>", unsafe_allow_html=True)

    st.write(
        "<div style='display: flex; justify-content: center;'>"
        "<img src='https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png' style='width: 150px;'>"
        "</div>",
        unsafe_allow_html=True
    )

    st.title("PharmaCare")
    choice = st.radio("Navigation", ["Extract text from images", "Display labeled text", "Download PDF Summary", "Fraud Detection"], index=0)
    st.info("This project application helps you annotate your medicine data and detect fraud.")
    st.sidebar.success("Select an option above.")


#-----------------------------------------RADIO BUTTON CHOICES--------------------------------------

#define containers for each choice's content
if choice == "Extract text from images":

        choice = st.radio('Choose an option', ["Upload Image", "Capture Image with Camera"])
        if choice == "Capture Image with Camera":
            st.write("# See a new image every second")
            controls = st.checkbox("Show controls") # button to show controls to pause the video (stop in one frame)
            image = camera_input_live(show_controls=controls) #pause the frame
            if image is not None:
                #st.write(type(image))
                st.image(image)  #show the image in the ui
                if st.button("Extract Text From Image"):
                    #convert the file to an opencv image.
                    pil_image = Image.open(image)
                    numpy_array = np.array(pil_image)
                    #opencv_image = cv2.imdecode(numpy_array, 1)
                    opencv_image = numpy_array.copy()
                    
                    # extract text and process the result
                    text, result = ner_app.ocr_extraction(opencv_image)
                    ####################SESSION STATE############################
                    st.session_state = text
                    ####################SESSION STATE############################
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
                        
                          

        elif choice == "Upload Image":
            st.title("Upload Your Image")
            uploaded_file = st.file_uploader("Choose an image file", type=(["jpg", "png", "jpeg"]))
            if uploaded_file is not None :
                #convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                # display the uploaded image
                st.image(opencv_image, channels="BGR")

                if st.button("Extract Text From Image"):
                    #st.write(type(opencv_image))
                    # extract text and process the result
                    text, result = ner_app.ocr_extraction(opencv_image)
                    
                    #compare with the extracted text that we've corrected
                    name = uploaded_file.name
                    file_csv_path = "C:/fraud_platform_imen/data/ocr_results_correct2.csv"
                    df = pd.read_csv(file_csv_path)
                    for index, row in df.iterrows():
                            # Extract the file name from the "Image Path" column
                            image_name = str(row["Path"])
                            # Compare the extracted file name with the provided file name
                            if name == image_name:
                                    # Extract the corresponding text
                                    text = row["Extracted Text"]
                                    break
                    ####################SESSION STATE############################
                    st.session_state = text
                    ####################SESSION STATE############################
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



elif choice == "Display labeled text":
    st.title('Named Entity Recognition')
    #text = '"Tretinoin Gel USP 0.1% wlw A-Ret" Gel 0.1% wlwMENARINI20g"'
    ####################SESSION STATE############################
    text = st.session_state 
    ####################SESSION STATE############################
   
    if st.button("Perform Named Entity Recognition"):
                doc = ner_app.perform_named_entity_recognition(text)
                ####################SESSION STATE############################
                st.session_state = doc
                ####################SESSION STATE############################
                
                html = ner_app.display_doc(doc)
                if html is not None : 
                    html_string = f"<h3>{html}</h3>"
                    # display annotated text
                    st.markdown("Here's the Annotated text:")
                    st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                    styled_text = f"<div class='background-text'>{html_string}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)
                else : 
                    st.write('none')

elif choice == "Download PDF Summary":
    st.title("Download PDF Summary")
    if st.button("Generate PDF Summary"):
        doc = st.session_state
        #st.write(doc)
        detail = ner_app.details_dict(doc)
        #st.write(detail)
        ner_app.create_file_txt(detail)
        ner_app.create_summary_pdf("details.txt")            
        with open("SUMMARY.pdf", "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            # Embedding PDF in HTML
            pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.session_state = detail
            st.download_button("Download PDF Summary", data=pdf_bytes, file_name="SUMMARY.pdf", mime="application/pdf")
            

if choice == "Fraud Detection":
    st.title("Fraud Detection with Jaccard Similarity")
    st.write('Detection of fraud in the text extracted : \n')
    if st.button("Calculation of Maximum Jaccard Score"):
        ####################SESSION STATE############################
        detail = st.session_state
        ####################SESSION STATE############################
        
        max_jaccard_score, entities, fraud_status = ner_app.fraud(detail)
        # Display result
        st.write(
                    "<span style='font-weight: bold; font-size: 20px;'>Max Jaccard Similarity Score: </span>"
                    "<br>",
                    unsafe_allow_html=True
                )

        #dsplay max jaccard score
        st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
        styled_text = f"<div class='background-text'>{max_jaccard_score}</div>"
        st.markdown(styled_text, unsafe_allow_html=True)

        #display entities
        entities = pd.DataFrame(entities)
        #st.dataframe(entities)
        st.write(entities)

        if fraud_status == "This Drug is potentially fraudulent":
            #dsplay conclusion
            st.markdown(set_background_color("#FF0000", 'White'), unsafe_allow_html=True)
            styled_text = f"<div class='background-text'>{fraud_status}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)
        else: 
            st.markdown(set_background_color("#008000", 'White'), unsafe_allow_html=True)
            styled_text = f"<div class='background-text'>{fraud_status}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)