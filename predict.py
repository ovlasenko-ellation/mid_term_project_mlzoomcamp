import pickle

model_file = 'model.bin'
with open(model_file, 'rb') as f_in:
    dv, rf = pickle.load(f_in)

input = {
    'hour': 14,
    'minute': 00,
    'day_of_the_week': "Wednesday"
}

x = dv.transform([input])
x


y = rf.predict(x)
y


print('input: ', input)
print('predicted traffic: ', y)

