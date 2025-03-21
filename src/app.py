# import libraries
import streamlit as st
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier

# Import custom components
from components.header import display_header, display_context
# from components.context import display_context

def main():
    """Main function to run the Streamlit app."""
    display_header()
    display_context()

if __name__ == "__main__":
    main()


# TODO 
# Table of content

# TODO
# Data discovery and preprocessing 

