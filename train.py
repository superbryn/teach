import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

f = pickle.load(open('data.pickle','rb'))

data = np.asarray(f['data'])
labels = np.asarray(f['labels'])

xTrain , xTest , yTrain , yTest = train_test_split(data, labels, test_size = 0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(xTrain,yTrain)

yPredict= model.predict(xTest)

score = accuracy_score(yPredict, yTest)

print(f"{score*100}% of samples were classfied correctly")


with open('model.p','wb') as f:
    pickle.dump({'model':model},f)






'''
print(type(f['data']))
print(len(f['data']))
# Filter out elements that do not have the shape (42,)
valid_data = [item for item in f['data'] if np.shape(item) == (42,)]

# Convert the filtered data to a NumPy array

# Optionally, print the shape of a few elements:
for i in range(26):  # Check the first 5 elements
    print(f'Shape of element {i}: {np.shape(f["data"][i])}')
'''