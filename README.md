# Fraud-Detection-Project
An application for fraud detection in medicine packages and tablets.

## Steps 

- Performed Web Scraping with Beautiful Soup to gather a large  dataset of of images and information about medicine packages and tablets packages and tablets .
- Transformed the data and loaded it into an excel file with Power Query Editor to have the names and information of each package in a local dataset.
- Extracted text from images with OCR tools such as EasyOCR and Pytesseract from medicine packages and tablets to detect fraud, and Aapplied Named Entity Recognition tagging on extracted text to label the medicine by name, dosage, type and size, and trained a custom spacy model on the processed data to predict labels on new text.
- Extracted the labeled text in a csv file and used Jaccard Similarity scores to detect fraud between the information on the packaging and the tablets.
- Built an user friendly dashboard with Streamlit.

#### Here is the demo video as well as a marketing video for our application
[Click here](https://drive.google.com/drive/folders/1zMqsyNe_j2EM4W50iTcAAprh4q-bZBoB?usp=drive_link)

#### In this repo I will be also putting a report of all the work done.


