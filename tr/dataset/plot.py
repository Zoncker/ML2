from sklearn.tree import DecisionTreeClassifier
import  pandas as pd


# Сетка для визуализации
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


df = pd.read_csv('german.csv')

clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)

# Обучение дерева
clf_tree.fit(train_data, train_labels)

# Отображение разделяющей поверхности
xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='viridis')
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap='viridis', edgecolors='black', linewidth=1.5)
plt.show()