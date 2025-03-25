import streamlit as st

# Pop up content
about_message =  """
### About Banking MLOps Application

This application provides a quick overview on our best model quality and give you the opportunity to try it yourself ! 

### Need help?  Here's how to get in touch:

* **Email Support:** support@mlops-banking.com
* **FAQ:** mlops-banking.com
* **Documentation:** mlops-banking.com

Please include the following information when contacting support:
* A description of the problem you are experiencing.
* Screenshots (if applicable).
* The version of the application you are using.
"""

# about_message="# This is a header. This is an *extremely* cool app!"

def display_header():
    st.title("🏦 Banking MLOps : ")
    

def display_contributor():
    """Displays the contributors section of the app."""
    
    st.markdown("""
                
    ---

    ### 👩🏻‍💻 **Contributors** :
    - *Cyrena Ramdani*
    - *Yoav COHEN*
    - *Hoang Thuy Duong VU*
    - *Salma LAHBATI*  
                """)


# Page congig 
def page_config():
    st.set_page_config(
            page_title="Banking MLOps", 
            page_icon="🏦", 
            layout="centered", 
            initial_sidebar_state="auto", 
            menu_items={
        'About': f"{about_message}"
        }
    )