import requests

if __name__ == "__main__":
    url = "http://127.0.0.1:9696/predict"

    input = {
        'hour': 3,
        'minute': 00,
        'day_of_the_week': "Monday"
    }
    response = requests.post(url, json=input).json()

    print(response)