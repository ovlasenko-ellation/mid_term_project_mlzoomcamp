import requests

if __name__ == "__main__":
    host = 'traffic-prediction-env.eba-cp9c5m9b.us-west-2.elasticbeanstalk.com.'
    url = f"http://{host}/predict"

    input = {
        'hour': 21,
        'minute': 00,
        'day_of_the_week': "Monday"
    }
    response = requests.post(url, json=input).json()

    print(response)