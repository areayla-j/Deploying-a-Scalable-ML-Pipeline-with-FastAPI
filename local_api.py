import requests
BASE = "http://127.0.0.1:8000"

# GET
r = requests.get(BASE + "/")
print("GET / ->", r.status_code, r.json())

# POST
payload = {
    "age": 39,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
pr = requests.post(BASE + "/predict", json=payload)
print("POST /predict ->", pr.status_code, pr.json())
