import requests

url = "https://adult-income-api-6av7.onrender.com/predict"
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

res = requests.post(url, json=payload)
print("HTTP Status Code:", res.status_code)
print("Response Text:", res.text)
