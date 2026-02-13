# DevOps & MLOps Practical Exam Project

This project is a complete DevOps + MLOps pipeline for a House Price Prediction model.
It includes dataset tracking using DVC, model training with MLflow tracking, FastAPI deployment, Docker containerization, docker-compose setup, and GitHub Actions CI/CD automation.

---

## Project Workflow

1. **Dataset Management (DVC)**
   - Dataset (`data/data.csv`) is tracked using DVC to ensure reproducibility.
   - DVC maintains version history of dataset changes.

2. **Model Training**
   - The training script (`ml/train.py`) loads the dataset and trains a Linear Regression model.
   - Training parameters are read from `ml/params.yml`.

3. **Experiment Tracking (MLflow)**
   - MLflow tracks training experiments and logs:
     - Parameters (test_size, random_state)
     - Metrics (MAE, RMSE)

4. **Model Saving**
   - After training, the model is saved as:
     - `api/model.pkl`

5. **FastAPI Deployment**
   - FastAPI app (`api/app.py`) loads the trained model and provides a `/predict` endpoint.
   - Users send JSON input features and receive predicted house price.

6. **Email Alerts**
   - Email alerts are triggered when:
     - Input schema is invalid
     - Model throws an exception during prediction
   - SMTP credentials are securely stored in `.env`.

7. **Dockerization**
   - The project is containerized using a `Dockerfile`.
   - `docker-compose.yml` runs the FastAPI service.

8. **CI/CD with GitHub Actions**
   - GitHub Actions workflow automatically:
     - Installs dependencies
     - Runs training script
     - Builds Docker image
     - Runs a container test

---

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r api/requirements.txt
pip install pandas scikit-learn joblib mlflow python-dotenv pyyaml