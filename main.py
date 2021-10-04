import numpy as np
from getmfcc import get_mfcc
from os import listdir
from os.path import isfile, join
from rbf import RBF

data = []

kv = {
    0: "atras",
    1: "adelante",
    2: "izquierda",
    3: "derecha"
}

for k in kv:
    path = "dataset/" + kv[k]
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for i in onlyfiles:
        mfcc_data = get_mfcc(path + "/" + i, k)
        data.append(mfcc_data)
        print(i, k, kv[k], mfcc_data)

data = np.array(data)

train_y = data[0:, 0]
train_x = data[0:, 1:]

# print("train_y", train_y)

clase = 1

test_y = []
test_x = []

for k in kv:
    path = "dataset/tests/" + kv[k]
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for i in onlyfiles:
        test_y.append(k)
        test_x.append(get_mfcc(path + "/" + i, k)[1:])

# test_y = [clase]
# test_x = [ get_mfcc("dataset/" + kv[clase] + "/adelante10.wav", clase)[1:] ]
# test_y = train_y
# test_x = train_x

RBF_CLASSIFIER = RBF(train_x, train_y, test_x, test_y, num_of_classes=4,
                     k=12, std_from_clusters=False)

matcheado = RBF_CLASSIFIER.fit()

print(test_y)
print(test_x)

# print("Matcheo con", matcheado, " => ", kv[matcheado[0]])
# print("Prueba realizada con: ", clase, " => ", kv[clase])
