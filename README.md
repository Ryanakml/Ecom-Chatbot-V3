# Production-Ready Ecommerce Chatbot

This repository contains the code and infrastructure for building a production-grade, AI-powered chatbot for ecommerce. The system is designed using MLOps principles for scalability, reproducibility, and maintainability.

## Architecture

*Diagrams to be added later.*

---

## Initial Setup

This project uses Docker Compose to manage local development infrastructure.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    It is recommended to use a Conda environment.
    ```bash
    conda env create -f environment.yml
    conda activate ecommerce-chatbot
    ```
    Alternatively, you can use `pip` with a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Launch local infrastructure:**
    This command will start Postgres, MinIO, and MLflow services in the background.
    ```bash
    docker-compose up -d
    ```

4.  **Verify services:**
    - **MinIO Console (Object Storage):** [http://localhost:9001](http://localhost:9001)
    - **MLflow UI (Experiment Tracking):** [http://localhost:5001](http://localhost:5001)
    - **Postgres (Database):** Accessible at `localhost:5432`

---