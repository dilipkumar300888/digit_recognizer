from sklearn import tree

features = [[150,0],[170,0],[140,1],[130,1]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

prediction = clf.predict([[160,0]])

if prediction[0] == 0:
    print("It's an Apple")
else:
    print("It's an Orange")