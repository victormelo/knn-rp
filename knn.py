#encoding: utf-8

import numpy as np
from scipy.io.arff import loadarff

def knn(Xt, Y, X, k=3):
    # input: Xt, amostras a serem classificadas
    #        Y, Rótulo das classes das amostras X de treinamento
    #        X, Amostras de treinamento
    #        k, n. de vizinhos do classificador k-NN
    # output: Ytest, labels previstas para as amostras Xt
    h, w = Xt.shape
    Ytest = np.empty((h,), '|S15')

    # para todos as h amostras de teste
    for i in range(h):

        # calculo da distância euclidiana de cada amostra de treino X
        # com a i amostra de teste
        difference = (X - Xt[i]) ** 2
        dist = np.sqrt(np.sum(difference, axis=1))

        # os k indices com menor distância
        k_indices = np.argsort(dist)[:k]
        labels = list(Y[k_indices])
        mode = max(labels, key=labels.count)
        Ytest[i] = mode


    return Ytest


def convert_to_numpy_array(list_of_tuples):
    array = list()
    for l in list_of_tuples:
        l = list(l);
        # remove a classe considerando que sempre estará na última coluna
        no_class = l[:-1]

        array.append(no_class)

    return np.array(array)


if __name__ == '__main__':

    train, meta = loadarff(open('vehicle-train.arff', 'r'))
    test, meta = loadarff(open('vehicle-test.arff', 'r'))

    # amostras de treinamento
    X = convert_to_numpy_array(train)

    # label a que pertence cada amostra de treinamento
    Y = train['class']

    # amostras de teste
    Xt = convert_to_numpy_array(test)

    # label que pertece as amostras de teste, usada para calcular a accuracy
    Yt = test['class']

    # myYt_k3 são as classes previstas para k=3
    myYt = knn(Xt, Y, X)

    certos_k3 = (myYt == Yt).sum()
    print 'Correct classified %d\n' % certos_k3
    print 'Accuracy k=3: %.4f%%' % (certos_k3 / float(myYt.size) * 100)

