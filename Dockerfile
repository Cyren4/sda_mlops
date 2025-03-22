# Use a minimal base image
FROM continuumio/miniconda3

# Create a directory for the application
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Install conda dependencies
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "banking-mlops", "/bin/bash", "-c"]
# RUN conda activate banking-mlops 

# Copy the application code
COPY . /app/

# Expose the port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
# CMD ["streamlit", "run", "src/app.py"]

# Run the Streamlit app using conda run to ensure the environment is active
CMD ["conda", "run", "-n", "banking-mlops", "streamlit", "run", "src/app.py"]
