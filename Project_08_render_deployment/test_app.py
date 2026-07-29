from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    print("Health Check Response:", res.json())
    assert res.status_code == 200

def test_predict():
    payload = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 22245,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Outlying-US(Guam-USVI-etc)"
    }
    res = client.post("/predict", json=payload)
    print("Predict Endpoint Response:", res.json())
    assert res.status_code == 200

if __name__ == "__main__":
    test_health()
    test_predict()
