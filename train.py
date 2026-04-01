# --- This is the training script for the models used in the web app ---
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('titanic_data.csv')
X = data[['Pclass', 'Age', 'Fare']]
y = data['Survived']

# Machine Learning: Ensemble Model (Voting Classifier)
m1 = LogisticRegression()
m2 = RandomForestClassifier()
m3 = SVC(probability=True)
model_ml = VotingClassifier(estimators=[('lr', m1), ('rf', m2), ('svc', m3)], voting='soft')
model_ml.fit(X, y)

# Neural Network: Multi-layer Perceptron (MLP)
model_nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
model_nn.fit(X, y)

#Saving the results (The .pkl files)
with open('model_ml.pkl', 'wb') as f:
    pickle.dump(model_ml, f)
with open('model_nn.pkl', 'wb') as f:
    pickle.dump(model_nn, f)

print("Training script executed successfully.")