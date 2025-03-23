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

4. Launch mlflow server and run all cells in the file `Mlflow.ipynb`
```shell
mlflow server --host 127.0.0.1 --port 8080
```


## Manual Build of Docker Image 

- Build the Docker image: 
```bash
docker build -t banking-mlops-app .
```

- Run the Docker image (NB: streamlit uses the port 8501 but we redirected to 8080)
```bash
docker run -p 8080:8080 banking-mlops-app
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
```bash
docker pull cramdani/sda_mlops_docker
```

## Manual Deployement to Cloud Run 

- Set up environment variable : 
```bash
export PROJECT_ID=
export REGION=europe-west1
```

- Connect to GCP and to the right project:
```bash
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud auth configure-docker $REGION-docker.pkg.dev

```

- Build and push the Docker image to Artifact Registry:
```bash
gcloud builds submit --tag  $REGION-docker.pkg.dev/$PROJECT_ID/banking-mlops/mlops-app:latest
```

- Deploy the Docker image to Cloud Run:
```bash
gcloud run deploy banking-mlops --image $REGION-docker.pkg.dev/$PROJECT_ID/banking-mlops/mlops-app:latest --region $REGION
```

## Automatic Deployement to Cloud Run 

1. Create Service Account on GCP
-   **banking-mlops-sa** - banking-mlops-sa@appmod-demo-lvl.iam.gserviceaccount.com

2. Add private key for service account (from console or cli) it will give permission to Github Action to push new services to GCP : 
```bash
gcloud iam service-accounts keys create
```

3. Create Secrets in Github Action :
- Service Account key : ```GCP_SA_KEY```
- Project ID : ```GCP_PROJECT_ID```
- Region : ```GCP_REGION```
- Service Name : ```GCP_SERVICE_NAME```

4. Create Github Action pipeline in **.github/workflows/gcp-cicd-deployement.yml**  

5. Push modification in the main branch
 

## File structure 
```
├── README.md
├── .github/workflows
│   └── github-cicd.py
├── data
│   ├── feature_importances.csv
│   ├── Loan_Data.csv
│   └── Loan_Data_Preprocessed.csv
├── environment.yml
├── src
│   ├── app.py
│   ├── components
│   │   ├── header.py
│   │   ├── introduction.py
│   │   ├── lstm.py
│   │   ├── page.py
│   │   └── random_forest.py
│   └── models
│       ├── Mlflow.ipynb
│       └── Preprocessing.ipynb
├── .dockerignore.ipynb
└── .gitignore.ipynb
```

## Contributors 
- Cyrena Ramdani
- Yoav COHEN
- Hoang Thuy Duong VU
- Salma LAHBATI



This project is a student project fulfilling the requirements of a MLOps Course.


## Source documentation
- [Run conda in Dockerfile](https://pythonspeed.com/articles/activate-conda-dockerfile/) 
- [Identity federation on gcp & github action](https://github.com/google-github-actions/auth#preferred-direct-workload-identity-federation)