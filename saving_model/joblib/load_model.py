import joblib

model = joblib.load('dlb_joblib.pkl')

output = model.predict([[1,2,3,4,5,6,7,8]])

print(output)