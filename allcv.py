from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn import datasets
from sklearn import svm

iris = datasets.load_wine()
X = iris.data
y = iris.target


kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):

    data_train = X[train_index]
    target_train = y[train_index]

    data_test = X[test_index]
    target_test = y[test_index]

    clf = svm.SVC(kernel='linear', C=1).fit(data_train, target_train)
    clf.score(data_test, target_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=kf)


print("KFold Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X):

    data_train = X[train_index]
    target_train = y[train_index]

    data_test = X[test_index]
    target_test = y[test_index]

    clf = svm.SVC(kernel='linear', C=1).fit(data_train, target_train)
    clf.score(data_test, target_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=loo)


print("LOO Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(X)

for train_index, test_index in rs.split(X):

    data_train = X[train_index]
    target_train = y[train_index]

    data_test = X[test_index]
    target_test = y[test_index]

    clf = svm.SVC(kernel='linear', C=1).fit(data_train, target_train)
    clf.score(data_test, target_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=rs)


print("ShuffleSplit Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)

for train_index, test_index in rkf.split(X):

    data_train = X[train_index]
    target_train = y[train_index]

    data_test = X[test_index]
    target_test = y[test_index]

    clf = svm.SVC(kernel='linear', C=1).fit(data_train, target_train)
    clf.score(data_test, target_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=rkf)


print("RepKFold Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

skf = LeavePOut(p=1)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):

    data_train = X[train_index]
    target_train = y[train_index]

    data_test = X[test_index]
    target_test = y[test_index]

    clf = svm.SVC(kernel='linear', C=1).fit(data_train, target_train)
    clf.score(data_test, target_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=skf)

print("CCV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))