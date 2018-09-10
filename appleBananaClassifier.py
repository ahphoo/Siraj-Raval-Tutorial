from sklearn import tree

#[length, width, height]
X = [[5,5,5],
	[5,6,6],
	[10,3,4],
	[6,7,7],
	[22,7,5],
	[12,12,11],
	[15,8,3],
	[6,1,2]]

Y = ["apple", "apple", "banana", "apple", "banana", "apple", "banana", "banana"]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

# Apple
prediction = clf.predict([[3,3,10]])

print(prediction)


