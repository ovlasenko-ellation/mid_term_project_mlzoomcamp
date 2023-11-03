import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

def load(model_file: str):
    with open(model_file, 'rb') as f_in:
         dv, rf = pickle.load(f_in)
         return dv, rf


app = Flask('get-traffic')


@app.route('/predict', methods=['POST'])

def predct(input):
    input = request.get_json()
    x = dv.transform([input])
    y = rf.predict(x)

    value_mapping = {0: 'low', 1: 'normal', 2: 'high', 3: 'heavy'}
    traffic_prediction = [value_mapping[value] for value in y]

    result = {
        'traffic_prediction': str(traffic_prediction)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

#My initial file
# model_file = 'model.bin'
# with open(model_file, 'rb') as f_in:
#     dv, rf = pickle.load(f_in)
#
# input = {
#     'hour': 14,
#     'minute': 00,
#     'day_of_the_week': "Wednesday"
# }
#
# def predct(input):
#     x = dv.transform([input])
#     y = rf.predict(x)
#     return y
#
#
# print('input: ', input)
#

