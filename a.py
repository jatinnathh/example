import requests

url = "http://192.168.1.34:8000/login"
data = {
    "email": "test",
    "password": "test123"
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
