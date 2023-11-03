import requests

if __name__ == "__main__":
    url = "http://127.0.0.1:8080/predict"

    input = {
        'hour': 14,
        'minute': 00,
        'day_of_the_week': "Wednesday"
    }
    response = requests.post(url, json=input).json()

    print(response)