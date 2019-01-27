import random
import datetime
import pickle
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt

def act(net):
    return np.where(net > 0, net, net * 0.5)

def dact(net):
    return np.where(net > 0, 1., 0.5)

def inv_act(h):
    return np.where(h > 0, h, h * 2)

def errfn(h2, ys):
    return 0.5 * np.sum(np.power(h2 - ys, 2))

def rand_onehots(shape):
    onehots = np.zeros(shape)
    cols = onehots.shape[1]
    for row in range(onehots.shape[0]):
        randcol = random.randint(0, cols-1)
        onehots[row, randcol] = 1.
    return onehots

def labels_to_onehots(labels, cardinality=100):
    res = np.zeros((len(labels), cardinality))
    for idx, label in enumerate(labels):
        res[idx][label] = 1.
    return res

def get_pairs():
    with open("cifar_train", "rb") as train_file:
        train_dict = pickle.load(train_file, encoding="bytes")
        train_labels = np.split(labels_to_onehots(train_dict[b"fine_labels"]), 1000)
        train_data = np.split(((train_dict[b"data"] / 256.) - 0.5), 1000)
        train_pairs = list(zip(train_data, train_labels))
    with open("cifar_test", "rb") as test_file:
        test_dict = pickle.load(test_file, encoding="bytes")
        test_labels = np.split(labels_to_onehots(test_dict[b"fine_labels"]), 500)
        test_data = np.split(((test_dict[b"data"] / 256.) - 0.5), 500)
        test_pairs = list(zip(test_data, test_labels))
    return train_pairs, test_pairs


if __name__ == "__main__":
    """
    Follows the notation of the bobdobbshess repo, exceptt s/J/err/g

    rank structure of w1, w2, xs, ys is not happy
    you basically have to have them _all_ be square and full rank
    """
    insize = 50
    hidsize = 50
    outsize = 100
    w1 = npr.randn(insize, hidsize) / (insize + hidsize)
    w2 = npr.randn(hidsize, outsize) / (hidsize + outsize)
    shaper = npr.randn(3072, insize) / np.sqrt(3072)
    temp_xs = npr.randn(50, 100)
    temp_ys = rand_onehots((50, 100))
    train_pairs, test_pairs = get_pairs()
    hess1 = np.zeros((insize * hidsize, insize * hidsize))
    hess2 = np.zeros((hidsize * outsize, hidsize * outsize))
    v1 = np.zeros_like(w1)
    v2 = np.zeros_like(w2)
    num_epochs = 200

    lamb = 1.
    alpha = 1e-1

    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        print(datetime.datetime.now())
        for it, pair in enumerate(train_pairs):
            xs, ys = pair
            shaped_xs = np.dot(xs, shaper)
            net1 = np.dot(shaped_xs, w1)
            h1 = act(net1)
            net2 = np.dot(h1, w2)
            h2 = act(net2)
            err = errfn(h2, ys)
            print("it: {}, err: {}".format(str(it), str(err)))
            print(datetime.datetime.now())
            if it % 50 == 0:
                corrects = 0.
                total = 0.
                for test_it, pair in enumerate(test_pairs):
                    xs, ys = pair
                    shaped_xs = np.dot(xs, shaper)
                    net1 = np.dot(shaped_xs, w1)
                    h1 = act(net1)
                    net2 = np.dot(h1, w2)
                    h2 = act(net2)
                    for idx in range(len(h2)):
                        pred = np.argmax(h2[idx])
                        correct = np.argmax(ys[idx])
                        if pred == correct:
                            corrects += 1
                        total += 1
                print("corrects: {}".format(corrects))
                print("total: {}".format(total))
            derr_dh2 = h2 - ys
            derr_dnet2 = derr_dh2 * dact(net2)
            derr_dh1 = np.dot(derr_dnet2, w2.T)
            derr_dnet1 = derr_dh1 * dact(net1)
            derr_dw2 = np.dot(h1.T, derr_dnet2)
            derr_dw1 = np.dot(shaped_xs.T, derr_dnet1)
            derr_dx = np.dot(derr_dnet1, w1.T)
            for idx, member in np.ndenumerate(w1):
                hessrow = (idx[0] * hidsize) + idx[1]
                v1[:] = 0.
                v2[:] = 0.
                v1[idx] = 1.
                analytic_rnet1 = np.dot(shaped_xs, v1)
                analytic_rh1 = analytic_rnet1 * dact(net1)
                analytic_rnet2 = np.dot(analytic_rh1, w2) + np.dot(h1, v2)
                analytic_rh2 = analytic_rnet2 * dact(net2)
                analytic_derr_dh2 = analytic_rh2
                analytic_derr_dnet2 = (analytic_derr_dh2 * dact(net2))
                analytic_derr_dh1 = np.dot(analytic_derr_dnet2, w2.T) + np.dot(derr_dnet2, v2.T)
                analytic_derr_dnet1 = (analytic_derr_dh1 * dact(net1))
                analytic_derr_dx = np.dot(analytic_derr_dnet1, w1.T) + np.dot(derr_dnet1, v1.T)
                analytic_derr_dw1 = np.dot(shaped_xs.T, analytic_derr_dnet1)
                hess1[hessrow, :] = analytic_derr_dw1.ravel()
            for idx, member in np.ndenumerate(w2):
                hessrow = (idx[0] * outsize) + idx[1]
                v1[:] = 0.
                v2[:] = 0.
                v2[idx] = 1.
                analytic_rnet1 = np.dot(shaped_xs, v1)
                analytic_rh1 = analytic_rnet1 * dact(net1)
                analytic_rnet2 = np.dot(analytic_rh1, w2) + np.dot(h1, v2)
                analytic_rh2 = analytic_rnet2 * dact(net2)
                analytic_derr_dh2 = analytic_rh2
                analytic_derr_dnet2 = (analytic_derr_dh2 * dact(net2))
                analytic_derr_dh1 = np.dot(analytic_derr_dnet2, w2.T) + np.dot(derr_dnet2, v2.T)
                analytic_derr_dnet1 = (analytic_derr_dh1 * dact(net1))
                analytic_derr_dx = np.dot(analytic_derr_dnet1, w1.T) + np.dot(derr_dnet1, v1.T)
                analytic_derr_dw2 = np.dot(h1.T, analytic_derr_dnet2) + np.dot(analytic_rh1.T, derr_dnet2)
                hess2[hessrow, :] = analytic_derr_dw2.ravel()
            hess1_eigval, hess1_eigvec = npl.eigh(hess1)
            hess2_eigval, hess2_eigvec = npl.eigh(hess2)
            w1 -= alpha * np.dot(np.dot(np.dot(hess1_eigvec, np.abs(np.diag(1. / (hess1_eigval + 0.1)))), hess1_eigvec.T), derr_dw1.ravel()).reshape(*w1.shape)
            w2 -= alpha * np.dot(np.dot(np.dot(hess2_eigvec, np.abs(np.diag(1. / (hess2_eigval + 0.1)))), hess2_eigvec.T), derr_dw2.ravel()).reshape(*w2.shape)
            # w1 -= alpha * np.dot(npl.inv(hess1), derr_dw1.ravel()).reshape(*w1.shape)
            # w2 -= alpha * np.dot(npl.inv(hess2), derr_dw2.ravel()).reshape(*w2.shape)
