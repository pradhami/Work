import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
name = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=name)

x = df.iloc[:,:-1]
y = df['class']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

# Find accuracy
result = model.score(x_test,y_test)
print(result)

# Save the model (.sav,.pkl)
pickle.dump(model, open('dlb_77.pkl','wb'))