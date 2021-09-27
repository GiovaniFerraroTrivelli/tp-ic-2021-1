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
        #print(i, k, kv[k], mfcc_data)

data = np.array(data)

train_y = data[0:13, 0]
train_x = data[0:13, 1:]

clase = 2

test_y = [clase]
test_x = [ get_mfcc("dataset/" + kv[clase] + "/Izquierda_3.wav", clase)[1:] ]

#test_y = [2]
#test_x = [[6.178471021141648, 8.742272395164237, -0.07535274698597458, -2.7496685455937517, 0.5430712171554284, 16.427636249127843, -6.852237165278303, -14.515578362576083, -5.962075487344079, -0.23589680712736577, -0.9092475820075999, -4.2154127876171446, 4.959226735068546]]

RBF_CLASSIFIER = RBF(train_x, train_y, test_x, test_y, num_of_classes=4,
                     k=4, std_from_clusters=False)

RBF_CLASSIFIER.fit()

print("Prueba realizada con: ", clase, " => ", kv[clase])