import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('cancer.csv')

# df.head()

df.diagnosis.value_counts()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

X = df.drop(['id', 'diagnosis', 'Unnamed: 32',], axis=1)
y = df['diagnosis']

scaler = MinMaxScaler()
mlps = []
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes = [10], activation ="tanh",alpha = 5, solver = 'sgd',learning_rate_init=0.2, max_iter = 15000,).fit(X_train_scaled, y_train)
mlps.append(mlp)

print("Training set score: %f" % mlp.score(X, y))
print("Training set loss: %f" % mlp.loss_)
print('Training score:', mlp.score(X_train_scaled, y_train))
print('Testing score:', mlp.score(X_test_scaled, y_test))
# plt.plot(mlp.loss_curve_)
# plt.plot(mlp.validation_scores_)


