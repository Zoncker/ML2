from numpy import array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import pandas as pd


def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return len(dataset[n*(i-1)//k:n*i//k])


def main():

    FOLD_I = 1
    FOLD_K = 10
    iris = load_iris()
    dataset = iris.data
    cl = iris.target
    df = pd.DataFrame({'C1': dataset[:, 0], 'C2': dataset[:, 1], 'C3': dataset[:, 2], 'C4': dataset[:, 3], 'Target': cl})
    print(df)
    counter = 1
    s = 0
    total_ac = 0
    while counter != FOLD_K+1:
        print("Fold ", counter)
        fold = fold_i_of_k(df, counter, 10)
        d_test = df[s:s + fold]
        X_test = d_test.iloc[:, 0:4]
        y_test = d_test.iloc[:, 4:5]

        d_train = df.drop(df.index[s: s + fold])
        X_train = d_train.iloc[:, 0:4]
        y_train = d_train.iloc[:, 4:5]

        X_train = X_train.values
        y_train = y_train.values

        X_test = X_test.values
        y_test = y_test.values
        y_train = array(y_train)

        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        ac = accuracy_score(y_test, lr_pred)
        print(ac)
        print(classification_report(y_test,lr_pred))

        total_ac = total_ac + ac
        s = s + fold
        counter = counter+1

    total_ac = total_ac / FOLD_K
    print("Cross validation accuracy is: ", total_ac)


main()
