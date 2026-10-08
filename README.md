# Speaker Authentication Using GRU and SVM

A machine learning project for speaker authentication, developed as part of a university machine learning and model deployment project.

The project explores whether a machine learning model can determine whether a speaker is an **authenticated user** or a **stranger** based on speech-derived Linear Predictive Coding (LPC) features. Two different classification approaches were implemented and compared: a **GRU neural network**, which models temporal dependencies in speech features, and an **SVM baseline**, which operates on flattened feature vectors.

The final project also includes a **REST API** built with FastAPI and packaged with Docker, allowing trained models to be deployed and queried through a simple HTTP endpoint.

---

## Project Overview

The original project was based on a 9-class speaker identification task. For this deployment project, the task was adapted into a **binary speaker authentication** problem:

- **Authenticated** — one of the recognised speakers
- **Stranger** — a speaker who is not authenticated

The dataset contains 9 speakers, with 3 speakers treated as authenticated users and 6 as strangers.

This change makes the problem more representative of a potential real-world security application, where the goal is not necessarily to identify exactly who is speaking, but rather to determine whether the speaker should be granted access.

The project compares two fundamentally different modelling approaches:

- **GRU (Gated Recurrent Unit)** — designed to capture temporal relationships between LPC feature frames.
- **SVM (Support Vector Machine)** — a traditional machine learning baseline using flattened feature vectors.

Both models use the same feature standardisation process to ensure a fair comparison.

---

## Dataset

The project uses the **Japanese Vowels dataset** from the UCI Machine Learning Repository.

The speech data is represented using Linear Predictive Coding (LPC) coefficients. Each time step contains:

- 12 LPC coefficients
- Up to 29 time frames

Therefore, an individual input can be represented as a maximum **29 × 12** time-series feature matrix.

For the SVM baseline, these features are flattened into a single **29 × 12 = 348-dimensional** feature vector.

The GRU instead receives the features as a sequence, allowing it to model temporal dependencies between consecutive frames.

---

## Machine Learning Models

### GRU

The GRU model builds on a pre-existing speaker-classification model and was adapted from the original 9-class identification task to the new binary authentication task.

The main advantage of the GRU approach is that it treats the LPC coefficients as a time series rather than simply as an unordered collection of features. This allows the model to learn temporal patterns in the speech signal.

The training pipeline is implemented in:

```
train_final_gru.py
```

### SVM Baseline

An SVM was implemented as a traditional machine learning baseline.

The SVM uses:

- An RBF kernel
- Flattened 348-dimensional feature vectors
- Grid search for hyperparameter selection
- Stratified 5-fold cross-validation

Unlike the GRU, the SVM does not explicitly model temporal dependencies. Flattening the input converts the complete sequence into a single feature vector.

The SVM training pipeline is implemented in:

```
train_svm.py
```

Comparing the two models provides insight into the trade-off between a sequential neural network and a conventional machine learning classifier.

---

## Preprocessing

The input LPC features are rescaled and standardised using **Z-score standardisation**.

The same standardisation approach is used for both the GRU and SVM to ensure that their results are directly comparable.

For API requests, preprocessing is handled automatically by the API:

- Input LPC sequences can contain between 1 and 29 time steps
- Sequences are padded to 29 frames
- Z-score standardisation is applied internally

This means API users only need to provide the raw LPC coefficient values.

---

## Class Balancing

Because the authentication task contains:

- 3 authenticated speakers
- 6 stranger speakers

the classes are imbalanced.

The GRU and SVM pipelines therefore incorporate class balancing so that the models do not simply favour the majority stranger class.

This is particularly important for an authentication application, where a model achieving high accuracy by predominantly predicting "stranger" would not necessarily be useful.

---

## Evaluation

The models were evaluated using several metrics rather than accuracy alone:

- Accuracy
- F1 score
- AUROC
- Calibration

Using multiple metrics provides a more complete view of model performance, particularly for the imbalanced binary authentication problem.

Both the SVM and GRU performed better than random classification according to the project evaluation.

The original 9-class GRU achieved a weighted F1 score of **0.94** before the task was adapted to binary authentication.

For the binary authentication task, the results are compared against:

- A majority-class baseline of **66.7%**
- A uniform random baseline of **50%**

---

## Project Structure

The main components of the project are:

```
.
├── src/
│   └── api/
│       └── app.py
│
├── saved_models/
│   └── trained model files
│
├── train_final_gru.py
├── train_svm.py
├── evaluate_final_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```

The exact repository structure may contain additional supporting files.

---

## Running the Project

### 1. Install Dependencies

All required dependencies are listed in:

```
requirements.txt
```

They can be installed using a standard Python package manager such as `pip` or `uv`.

The project requires:

- Python 3.11+
- PyTorch
- scikit-learn
- FastAPI
- Uvicorn
- joblib
- NumPy
- Pydantic

See `requirements.txt` for the exact dependency versions.

### 2. Train the Models

Run the model scripts in the following order:

```bash
python train_final_gru.py
python train_svm.py
python evaluate_final_model.py
```

The trained model files should be placed in `saved_models/` before starting the API.

---

## REST API

The trained authentication model is exposed through a FastAPI REST API.

The API accepts LPC coefficient time series and returns an authentication prediction together with confidence scores.

### Start the API

Make sure `saved_models/` contains the trained model files first. Then run:

```bash
uvicorn src.api.app:app
```

The API will be available at:

```
http://localhost:8000
```

Interactive Swagger API documentation is available at:

```
http://localhost:8000/docs
```

The Swagger interface can be used to test the API directly from a browser.

### Making a Prediction

Send a `POST` request to:

```
/predict
```

with a JSON body containing the LPC coefficient time series.

**Example request:**

```json
{
  "lpc_coefficients": [
    [-0.52, 0.18, 0.09, -0.31, 0.42, -0.12, 0.02, 0.28, -0.19, 0.48, -0.38, 0.11],
    [0.11, -0.21, 0.32, -0.09, -0.28, 0.19, 0.38, -0.47, 0.03, 0.31, -0.40, 0.22]
  ]
}
```

The API handles:

- Input validation
- Padding sequences to 29 frames
- Z-score standardisation
- Model inference
- Returning the authentication result and confidence scores

Input sequences can contain between 1 and 29 time steps.

**Example response:**

```json
{
  "authenticated": true,
  "confidence_authenticated": 0.97,
  "confidence_stranger": 0.03
}
```

A response of `"authenticated": true` indicates that the model classified the speaker as an authenticated speaker.

The confidence fields provide the model's predicted confidence for each class.

### Example Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lpc_coefficients": [
      [-0.5, 0.2, 0.1, -0.3, 0.4, -0.1, 0.0, 0.3, -0.2, 0.5, -0.4, 0.1]
    ]
  }'
```

---

## Docker Deployment

The API can also be packaged and run as a Docker container.

Build the Docker image:

```bash
docker build -t speaker-api .
```

Run the container:

```bash
docker run -p 8000:8000 speaker-api
```

The API can then be accessed at `http://localhost:8000` and the interactive documentation at `http://localhost:8000/docs`.

This provides a portable deployment of the trained speaker-authentication system without requiring the API environment to be configured manually on the host machine.

---

## Model Comparison

The project compares two different approaches to speaker authentication.

| Model | Feature Representation | Key Characteristic |
|-------|------------------------|--------------------|
| GRU | Sequential 29 × 12 LPC features | Models temporal dependencies |
| SVM | Flattened 348-dimensional vector | Traditional non-sequential baseline |

The SVM provides a useful baseline for determining whether explicitly modelling the temporal structure of the LPC features provides an advantage.

The GRU is better suited to the sequential nature of the input because it can learn relationships between successive time frames, whereas the flattened SVM representation loses this explicit temporal structure.

---

## Deployment Architecture

At a high level, the deployed system follows this pipeline:

```
LPC Coefficient Time Series
          │
          ▼
     FastAPI REST API
          │
          ▼
    Input Validation
          │
          ▼
 Padding + Standardisation
          │
          ▼
   Trained Authentication
          Model
          │
          ▼
 Authentication Prediction
          │
          ▼
     JSON Response
```

The API and its dependencies can be packaged into a Docker container:

```
Client
  │
  │ HTTP POST /predict
  ▼
Docker Container
  │
  ├── FastAPI
  ├── Preprocessing
  └── Trained ML Model
          │
          ▼
     JSON Prediction
```

---

## Technologies Used

- Python
- PyTorch
- scikit-learn
- FastAPI
- Uvicorn
- Docker
- NumPy
- Pydantic
- joblib

---

## References

- Japanese Vowels Dataset — UCI Machine Learning Repository
- FastAPI documentation
- PyTorch documentation
- scikit-learn documentation
