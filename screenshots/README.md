# Deploying a Scalable ML Pipeline with FastAPI

**Student:** Ar’eayla Jeanpierre  
**Course:** D501 – Machine Learning DevOps  


https://github.com/areayla-j/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions

## Deliverables

- **Model + Training Script**
  - `ml/model.py`, `train_model.py`

- **Unit Tests**
  - `test_ml.py` (3+ passing tests)

- **API**
  - `main.py` (includes GET + POST endpoints)
  - Tested locally with `local_api.py`

- **Artifacts**
  - `model/encoder.joblib`
  - `model/label_binarizer.joblib`

- **Slice Metrics**
  - `slice_output.txt` (performance by education feature)

- **Model Card**
  - `model_card.md` (all sections complete, personalized)

- **Screenshots in `screenshots/`**
  - `unit_test.png` – pytest passing  
  - `local_api.png` – GET + POST working  
  - `continuous_integration.png` – GitHub Actions passing  
  - `slice_output.png` – slice metrics  
  - `model_metrics.png` – precision/recall/F1 results  
  - `final_checks.png` – flake8 + pytest clean  

---

## Notes
This project demonstrates deploying a scalable machine learning pipeline with FastAPI, including:
- Training and saving a model
- Serving predictions via REST API
- Automated testing and CI/CD with GitHub Actions
- Model performance tracking and documentation

