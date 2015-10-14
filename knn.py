#encoding: utf-8

import numpy as np
from scipy.io.arff import loadarff
from collections import Counter


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
        difference = np.abs(
            X.astype(np.float64) - Xt[i].astype(np.float64)) ** 2
        dist = np.sqrt(np.sum(difference, axis=1))

        # os k indices com menor distância
        k_indices = np.argsort(dist)[:k]

        # coloca na amostra i o label que mais se repete dentro dos k labels
        [(label, times)] = Counter(Y[k_indices]).most_common(1)
        Ytest[i] = label

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

    train, meta = loadarff(open('iris-train.arff', 'r'))
    test, meta = loadarff(open('iris-test.arff', 'r'))

    # amostras de treinamento
    X = convert_to_numpy_array(train)

    # label a que pertence cada amostra de treinamento
    Y = train['class']

    # amostras de teste
    Xt = convert_to_numpy_array(test)

    # label que pertece as amostras de teste, usada para calcular a accuracy
    Yt = test['class']

    # myYt são as classes previstas
    myYt = knn(Xt, Y, X)

    certos = (myYt == Yt).sum()
    print 'Accuracy: %.4f%%' % (certos / float(myYt.size) * 100)
