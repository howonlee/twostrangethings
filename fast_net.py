import random
import datetime
import pickle
import numpy as np
import numpy.random as npr
import numpy.linalg as npl

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

def calc_hess():
    """ dead code for calculating analytic hessian. all of it """
    """
    hess = np.zeros((insize * outsize, insize* outsize))
    for idx, member in np.ndenumerate(w):
        hessrow = (idx[0] * outsize) + idx[1]
        v[:] = 0.
        v[idx] = 1.
        rnet = np.dot(xs, v)
        rh = rnet * dact(net)
        rderr_dh = rh
        # ddact doesn't exist
        rderr_dnet = rderr_dh * dact(net)
        rderr_dw = np.dot(xs.T, rderr_dnet)
        hess[hessrow, :] = rderr_dw.ravel()
    w -= alpha * np.dot(npl.inv(hess), derr_dw.ravel()).reshape(derr_dw.shape)
    """
    pass

def labels_to_onehots(labels, cardinality=100):
    res = np.zeros((len(labels), cardinality))
    for idx, label in enumerate(labels):
        res[idx][label] = 1.
    return res

def get_pairs():
    with open("cifar_train", "rb") as train_file:
        train_dict = pickle.load(train_file, encoding="bytes")
        train_labels = np.array_split(labels_to_onehots(train_dict[b"fine_labels"]), 20)
        train_data = np.array_split(((train_dict[b"data"] / 256.) - 0.5), 20)
        print(train_data[0].mean())
        train_pairs = list(zip(train_data, train_labels))
    with open("cifar_test", "rb") as test_file:
        test_dict = pickle.load(test_file, encoding="bytes")
        test_labels = np.split(labels_to_onehots(test_dict[b"fine_labels"]), 500)
        test_data = np.split(((test_dict[b"data"] / 256.) - 0.5), 500)
        test_pairs = list(zip(test_data, test_labels))
    return train_pairs, test_pairs

if __name__ == "__main__":
    insize = 3072
    outsize = 100
    w = npr.randn(insize, outsize) / (insize + outsize)
    train_pairs, test_pairs = get_pairs()
    num_epochs = 10
    alpha = 1e-1
    r = 1e-9
    """
    This implementation is done w/ fast fd newton's
    I haven't figured out fast fd newton's for nontrivial net yet
    """

    for epoch in range(num_epochs):
        # random.shuffle(train_pairs)
        print("epoch: {}".format(str(epoch)))
        for it, pair in enumerate(train_pairs):
            xs, ys = pair
            net = np.dot(xs, w)
            h = act(net)
            err = errfn(h, ys)
            if it % 1 == 0:
                print("it: {}, err: {}".format(str(it), str(err)))
            derr_dh = h - ys
            derr_dnet = derr_dh * dact(net)
            derr_dw = np.dot(xs.T, derr_dnet)
            # w -= alpha * derr_dw
            fd1_derr_dw = derr_dw * (1. + (r / 2.))
            fd2_derr_dw = derr_dw * (1. - (r / 2.))
            print("start first pinv")
            fd1_derr_dnet = np.dot(npl.pinv(xs.T), fd1_derr_dw)
            fd2_derr_dnet = np.dot(npl.pinv(xs.T), fd2_derr_dw)
            print("end first pinv")
            fd1_derr_dh = fd1_derr_dnet / dact(net)
            fd2_derr_dh = fd2_derr_dnet / dact(net)
            fd1_h = fd1_derr_dh + ys
            fd2_h = fd2_derr_dh + ys
            fd1_net = inv_act(fd1_h)
            fd2_net = inv_act(fd2_h)
            print("start second pinv")
            fd1_w = np.dot(npl.pinv(xs), fd1_net)
            fd2_w = np.dot(npl.pinv(xs), fd2_net)
            print("end second pinv")
            fd_w = (fd1_w - fd2_w) / r
            w -= alpha * fd_w
        corrects = 0.
        total = 0.
        for it, pair in enumerate(test_pairs):
            xs, ys = pair
            net = np.dot(xs, w)
            h = act(net)
            for idx in range(len(h)):
                pred = np.argmax(h[idx])
                correct = np.argmax(ys[idx])
                if pred == correct:
                    corrects += 1
                total += 1
        print("corrects: {}".format(corrects))
        print("total: {}".format(total))
