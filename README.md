# Banking MLOps : Predicting Loan Defaults in Retail Banking

**Context:**

We are a new team in the retail banking sector, which is currently experiencing higher-than-expected default rates on personal loans. Personal loans are a significant source of revenue for banks, but they carry the inherent risk that borrowers may default. A default occurs when a borrower stops making the required payments on a debt.

**Objective:**

The risk team is analyzing the existing loan portfolio to forecast potential future defaults and estimate the expected loss. The primary goal is to build a predictive model that estimates the probability of default for each customer based on their characteristics. Accurate predictions will enable the bank to allocate sufficient capital to cover potential losses, thereby maintaining financial stability.


## Set-up Project 

1.  Clone the repository:

```bash
git clone https://github.com/Cyren4/mlops_cours.git
cd mlops_cours
```

2.  Activate virtual env
```shell
conda env create -f environment.yml
conda activate banking-mlops
```

3.  Launch streamlit application : 
```
streamlit run src/app.py
```

4. Launch mlflow server and run all cells in the file `mlflow_test.ipynb`
```shell
mlflow server --host 127.0.0.1 --port 8080
```


## Manual Deployement 

- Build the Docker image: 
```bash
docker build -t banking-mlops-app .
```

- Run the Docker image (NB: streamlit uses the port 8501)
```bash
docker run -p 8501:8501 banking-mlops-app
```

Comments : This maps port 8501 on your host machine to port 8501 in the container (the port Streamlit uses). You should then be able to access your Streamlit app in your browser at ``http://localhost:8501``.

- Push this image to dockerhub: 
```bash
docker login 
docker image tag banking-mlops:latest cramdani/sda_mlops_docker
docker push cramdani/sda_mlops_docker  
```

- Repo to track the images : [cramdani/sda_mlops_docker](https://hub.docker.com/r/cramdani/sda_mlops_docker) 

- Get its images  : 
```shell
docker pull cramdani/sda_mlops_docker
```

- Streamlit application 


## File structure 
```
├── README.md
├── assets
│   └── __init__.py
├── bin
│   └── dataloader.py
├── config
│   └── config.yml
├── data
│   ├── Loan_Data.csv
│   └── Loan_Data_Preprocessed.csv
├── environment.yml
├── src
│   ├── app.py
│   ├── components
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.py
│   │   │   └── header.py
│   │   └── header.py
│   └── models
│       ├── Mlflow.ipynb
│       ├── Preprocessing.ipynb
│       └── __init__.py
└── vis_data.ipynb
```

## Contributors 
- Cyrena Ramdani
- Yoav COHEN
- Hoang Thuy Duong VU
- Salma LAHBATI



This project is a student project fulfilling the requirements of a MLOps Course.


## Source documentation
- [Run conda in Dockerfile](https://pythonspeed.com/articles/activate-conda-dockerfile/) 