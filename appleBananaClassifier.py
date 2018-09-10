from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

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
prediction = clf.predict([[100,5,12]])

print(prediction)

gnb = GaussianNB()
gnb = gnb.fit(X, Y)

prediction = gnb.predict([[100,5,12]])

print(prediction)

clf = svm.SVC()
clf.fit(X, Y)
prediction = clf.predict([[100,5,12]])

print(prediction)




