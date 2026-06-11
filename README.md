**Model Implementation**:

Install Dependencies:

all dependencies included in requirements.txt -- can be installed using any normal process, such as with uv or pip

For Models: run in order
1. train_final_gru.py 
2. train_svm.py 
3. evaluate_final_model.py

**API Implementation**:
# Make sure saved_models/ contains the trained files first
uvicorn src.api.app:app 

The API will be available at http://localhost:8000. Open http://localhost:8000/docs for the interactive Swagger documentation.

### Docker

Build the image and run the container:

```bash
docker build -t speaker-api .
docker run -p 8000:8000 speaker-api

## Making a prediction
Send a POST request to /predict with a JSON body containing the LPC coefficient time series:
```
{
  "lpc_coefficients": [
    [-0.52, 0.18, 0.09, -0.31, 0.42, -0.12, 0.02, 0.28, -0.19, 0.48, -0.38, 0.11],
    [0.11, -0.21, 0.32, -0.09, -0.28, 0.19, 0.38, -0.47, 0.03, 0.31, -0.40, 0.22]
  ]
}
```
The API handles padding to 29 frames and Z-score standardization internally — you only need to send the raw LPC values. Sequences can be any length between 1 and 29 time steps.

### Example response
```
{
  "authenticated": true,
  "confidence_authenticated": 0.97,
  "confidence_stranger": 0.03
}
```
### Example using curl
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lpc_coefficients": [[-0.5, 0.2, 0.1, -0.3, 0.4, -0.1, 0.0, 0.3, -0.2, 0.5, -0.4, 0.1]]}'
```
## Requirements

- Python 3.11+

- PyTorch

- scikit-learn

- FastAPI

- uvicorn

- joblib

- numpy

- pydantic

See requirements.txt for exact versions.
## Baseline comparison
The SVM baseline (train_svm.py) uses an RBF kernel on flattened 348-dimensional feature vectors (12 LPC coefficients x 29 time steps). The GRU model outperforms this baseline by modelling temporal dependencies that the flattened representation loses.
## Results
The original 9-class GRU achieved a weighted F1 of 0.94. The binary authentication version significantly exceeds the majority-class baseline of 66.7% and the uniform random baseline of 50%.
## References

- Japanese Vowels dataset: UCI Machine Learning Repository

```

```
