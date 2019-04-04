from keras.layers import PReLU, Conv2D, Concatenate, SpatialDropout2D


class PRELU(PReLU):
    """
    this class is to fix bug with saving/loading model weights
    """
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)


def conv_prelu(n=128, kernel_size=(2, 2), padding='valid'):
    def inside(x):
        x = Conv2D(filters=n, kernel_size=kernel_size, padding=padding)(x)
        x = SpatialDropout2D(.5)(x)
        x = PRELU()(x)
        return x
    return inside


def vh_concat(n=128, kernel_size=2, padding='valid'):
    def inside(x):
        x1 = conv_prelu(n=n, kernel_size=(kernel_size, 1), padding=padding)(x)
        x2 = conv_prelu(n=n, kernel_size=(1, kernel_size), padding=padding)(x)
        x3 = Concatenate()([x1, x2])
        return x3

    return inside


