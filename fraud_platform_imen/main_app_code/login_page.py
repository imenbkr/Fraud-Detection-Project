import streamlit as st
import ner_app  # Import the functions from the ner_app.py file
import mysql.connector
import bcrypt
import numpy as np
import pandas as pd

#import session_state
#from session_state import SessionState
import base64
from streamlit_extras.switch_page_button import switch_page
#from streamlit_option_menu import option_menu

# MySQL Connection
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="fraud_accounts"
)
#-----------------------------------------LOGING AND VERIFICATION--------------------------------------
def check_password(provided_password, stored_password_hash):
    return bcrypt.checkpw(provided_password, stored_password_hash)

def to_bytes(s):
    if type(s) is bytes:
        return s
    elif type(s) is str or (sys.version_info[0] < 3 and type(s) is unicode):
        return codecs.encode(s, 'utf-8')
    else:
        raise TypeError("Expected bytes or string, but got %s." % type(s))

def validate_login(email, password):
    cursor = db.cursor()
    query = "SELECT password FROM users WHERE email = %s"
    cursor.execute(query, (email,))
    stored_password_hash = cursor.fetchone()

    if stored_password_hash and bcrypt.checkpw(to_bytes(password), bytes(stored_password_hash[0])):
        return True
    return False


#CREATE AN ACCOUNT FUNCTION
def create_account(email, password):
    # Hash the provided password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the user's email and hashed password into the database
    cursor = db.cursor()
    query = "INSERT INTO users (email, password) VALUES (%s, %s)"
    cursor.execute(query, (email, hashed_password))
    db.commit()

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



#-----------------------------------------SESSION STATE--------------------------------------
# Initialize the session state
#session_state = SessionState()
# Access the values using the get method
#current_page = session_state.get_current_page
#user_logged_in = session_state.get_user_logged_in

#-----------------------------------------LOGIN PAGE--------------------------------------
# Page function for the login page
#def login(temp):
st.title("Medicine Fraud Detection App")
set_background('C:/fraud_platform_imen/widgets/medicine-capsules.png')
st.write("Welcome! Please log in or sign up to continue:")
    
# Provide unique keys for the radio buttons
option = st.radio("Select Option", ["Login", "Sign Up"], key="login_radio")
temp=False
if option == "Sign Up":
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            create_account(new_email, new_password)
            st.success("Account created successfully! Now you can log in!")

elif option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Log In"):
            if validate_login(email, password.encode('utf-8')):
                st.success("Login Successful!")
                temp = True
                switch_page("main")  # Switch to the welcome page
            st.warning('Verify your information!')
                
        if temp : 
           pass

           # st.warning('You have to login first!')


#if __name__ == "__main__":
#    main()