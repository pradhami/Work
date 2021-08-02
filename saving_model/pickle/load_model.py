import pickle

model = pickle.load(open('dlb_77.pkl','rb'))

output = model.predict([[1,2,3,4,5,6,7,8]])

if(output == 1):
    print('Diabetic')
else:
    print('Not Diabetic')