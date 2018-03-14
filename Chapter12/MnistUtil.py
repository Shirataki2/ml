import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    # labels_path = ('mnist\\%s-labels.idx1-ubyte' % kind)
    # images_path = ('mnist\\%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        _1, _2 = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        _3, _4, _5, _6 = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


if __name__ == '__main__':
    X_train, y_train = load_mnist('mnist', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    fig, ax = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True)
    ax = ax.flatten()
    for i, j in enumerate([1, 1, 4, 5, 1, 4]):
        img = X_train[y_train == j][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
