import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\Users\HP\Downloads\kag_risk_factors_cervical_cancer.csv")
df.head() ##this returns the first five rows of the dataset

df.shape ##returns the no. of rows and columns
df.dtypes
df.describe
df.isnull().sum() ##check for null values
df.duplicated().any() ##check for duplicate values

features = df[['Age', 'Number of sexual partners', 'First Sexual intercourse', 'Smoke(years)', 'Smokes(packs/years)', 'Smokes (packs/year)', 'IUD', 'IUD (years)', 'STDs',
       'STDs (number)', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology',
       'Biopsy']]
X = df[['Age', 'Number of sexual partners', 'First Sexual intercourse', 'Smoke(years)', 'Smokes(packs/years)', 'Smokes (packs/year)', 'IUD', 'IUD (years)', 'STDs',
       'STDs (number)', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']]
y = df['Biopsy']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    display(X_test)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
from sklearn import model_selection, metrics
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score
prediction = model.predict(X_test) 
prediction
accuracy = metrics.accuracy_score(y_test,prediction)
accuracy

import pickle
pickle.dump(RandomForestClassifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model)


