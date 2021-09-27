import numpy as np


def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k, max_iters):
    # Aleatoriamente, se seleccionan K centroides del dataset dado (estado inicial)
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    converged = False

    current_iter = 0

    while (not converged) and (current_iter < max_iters):

        # Genera una matriz con tantas filas como centroides haya
        cluster_list = [[] for i in range(len(centroids))]

        for x in X:  # Go through each data point
            distances_list = []
            # Calcula las distancias del punto x a todos los centroides
            for c in centroids:
                distances_list.append(get_distance(c, x))

            # Asigna cada punto al cluster (grupo) de puntos que tiene menor distancia al centroide
            cluster_list[int(np.argmin(distances_list))].append(x)

        # Filtra los 0 (los centroides)
        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []

        # Calcula el nuevo centroide de la iteracion actual
        for j in range(len(cluster_list)):
            # El centroide nuevo se calcula como la media de los elementos del cluster j
            centroids.append(np.mean(cluster_list[j], axis=0))

        # Calcula el error segun corregido de la iteracion anterior
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        print('K-MEANS: ', int(pattern))

        converged = (pattern == 0)

        current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]


class RBF:

    def __init__(self, X, y, tX, ty, num_of_classes,
                 k, std_from_clusters=True):
        self.X = X
        self.y = y

        self.tX = tX
        self.ty = ty

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    # Devuelve el valor de la funcion de activacion dado un punto x, un centroide c y una desviacion estandar s
    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        # Funcion de activacion vieja
        # return 1 / np.exp(-distance / s ** 2)
        return np.exp(-distance / s ** 2)

    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        # Devuelve todos los resultados de aplicar cada entrada a todos los perceptrones
        for x in X:
            # Agrega a la lista la funcion de activacion de cada perceptron, evaluada en x
            RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])

        return np.array(RBF_list)

    def test(self, X):
        distances = self.rbf_list([ X ], self.centroids, self.std_list)
        result = distances @ self.w
        return np.argmax(result)

    def fit(self):

        # ENTRENAMIENTO

        self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)

        # Lo utiliza para lacular la desviacion estandar
        if not self.std_from_clusters:
            # Calcula la maxima distancia entre 2 centroides cualesquiera
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            # Crea una lista de k elementos que contienen la desviacion estandar (mismo valor) de cada perceptron
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)

        # Calcula el peso lineal de cada perceptron
        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)

        # EJECUCION

        RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)

        self.pred_ty = RBF_list_tst @ self.w

        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        diff = self.pred_ty - self.ty

        print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
