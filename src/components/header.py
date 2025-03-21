import streamlit as st

def display_header():
    """Displays the header section of the app."""
    st.title("Banking MLOps")
    st.subheader("Predicting Loan Defaults in Retail Banking")
    st.write("""Contributors :
- **Cyrena Ramdani**
- **Yoav COHEN**
- **Hoang Thuy Duong VU**
- **Salma LAHBATI**
""")

# App description of context and goals 
def display_context():
    """Displays the context and objective of the app."""
    st.markdown("""
### **Context:**

We are a team in the retail banking sector, which is currently experiencing higher-than-expected default rates on personal loans. Personal loans are a significant source of revenue for banks, but they carry the inherent risk that borrowers may default. A default occurs when a borrower stops making the required payments on a debt.

### **Objective:**

The risk team is analyzing the existing loan portfolio to forecast potential future defaults and estimate the expected loss. The primary goal is to build a predictive model that estimates the probability of default for each customer based on their characteristics. Accurate predictions will enable the bank to allocate sufficient capital to cover potential losses, thereby maintaining financial stability.

""")
