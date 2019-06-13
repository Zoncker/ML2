from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
import time


def error_cnt(dataset, features):
    X = dataset.data[:, features]
    y = dataset.target
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    y_predict = clf.predict(X)
    return (dataset.target != y_predict).sum()


def add_selection(dataset, result=None):
    feature_count = len(dataset.data[0])
    if result is None:
        result = {
            "error": 1e9,
            "features": []
        }

    while True:
        current_result = result

        for feature in range(feature_count):
            if feature in result["features"]:
                continue

            features = result["features"] + [feature]
            features.sort()

            error = error_cnt(dataset, features)

            if error <= current_result["error"]:
                current_result = {
                    "error": error,
                    "features": features
                }

        if current_result == result:
            break

        result = current_result

    return result


def del_selection(dataset, result=None):
    feature_count = len(dataset.data[0])
    if result is None:
        features_all = list(range(feature_count))
        error = error_cnt(dataset, features_all)

        result = {
            "error": error,
            "features": features_all
        }

    while len(result["features"]) > 1:
        current_result = result

        for feature in result["features"]:
            features = result["features"][:]
            features.remove(feature)

            error = error_cnt(dataset, features)

            if error <= current_result["error"]:
                current_result = {
                    "error": error,
                    "features": features
                }

        if current_result == result:
            break

        result = current_result

    return result


def add_del_selection(dataset):
    result = {
        "error": 1e9,
        "features": []
    }

    while True:
        new_result = add_selection(dataset, result)
        new_result = del_selection(dataset, new_result)

        if new_result["error"] >= result["error"]:
            break

        result = new_result

    return result


dataset = datasets.load_iris()
feature_count = 13
dataset.data = dataset.data[:, :feature_count]


def run_time(algorithm):
    start = time.process_time()
    result = algorithm()
    elapsed = int((time.process_time() - start) * 1000)
    print("Время работы:", "{:,}".format(elapsed).replace(",", " "), "мс")
    return result


def quality(result):
    error = result["error"]
    total = len(dataset.data)
    quality = round((total - error) / total * 100, 2)
    print("Качество алгоритма:", str(quality) + "%")


def log_features(result):
    features = np.asarray(result["features"])
    names = dataset.feature_names
    print("Признаки: ", list(map(lambda i: str(i) + " – " + names[i], features)))


def log(message, algorithm):
    print(message)
    result = run_time(algorithm)
    print("Количество ошибок:", result["error"])
    quality(result)
    print("Количество признаков: ", len(result["features"]))
    log_features(result)
    print("")


log("Метод ADD", lambda: add_selection(dataset))

log("Метод DEL", lambda: del_selection(dataset))

log("Метод ADD-DEL", lambda: add_del_selection(dataset))