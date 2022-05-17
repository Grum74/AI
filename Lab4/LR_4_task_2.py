import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.cluster import KMeans

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
dataset = read_csv(url)

# Розділення датасету
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

plt.scatter(X[:, 0], X[:, 3], s=50)

# Параметри кластеризації
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                verbose=0, random_state=
                None, copy_x=True, algorithm='auto')
# Тренування
kmeans.fit(X)

# Передбачення найближчого кластеру
y_kmeans = kmeans.predict(X)

# Налаштування відображення
plt.scatter(X[:, 0], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 3], c='black', s=200, alpha=0.5)
plt.show()

# Оголошення функції
def find_clusters(X, n_clusters, rseed=2):
    # Випадковий вибір кластерів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Зіставити підписи на основі найближчого центру
        labels = pairwise_distances_argmin(X, centers)

        # Знайти нові центри на основі значень точок
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # Перевірка на зіставлення
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 3], c=labels,
            s=50, cmap='viridis')
plt.show()
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 3], c=labels,
            s=50, cmap='viridis')
plt.show()
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 3], c=labels,
            s=50, cmap='viridis')
plt.show()
