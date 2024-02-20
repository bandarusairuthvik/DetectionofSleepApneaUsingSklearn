import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import pickle

filename = "ApneaData.pkl"
test_percent = 0.2

t = time.time()
with open(filename, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)

features = np.array([row[:-1] for row in data])
classes = np.array([row[-1] for row in data])
input_length = len(features)
test_length = int(input_length * test_percent)

train_features, train_classes = features[:-test_length], classes[:-test_length]
test_features, test_classes = features[-test_length:], classes[-test_length:]

print("preprocessing time is:", (time.time() - t))
t = time.time()

clf = RandomForestClassifier(n_estimators=30)
clf.fit(train_features, train_classes)

print("fitting time is:", (time.time() - t))
t = time.time()

pred_classes = clf.predict(test_features)
score = accuracy_score(pred_classes, test_classes) * 100

print("predicting time is:", (time.time() - t))
print("Accuracy is:", score)