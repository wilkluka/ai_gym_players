from keras.layers import PReLU, Conv2D, Concatenate, DepthwiseConv2D


class PRELU(PReLU):
    """
    this class is to fix bug with saving/loading model weights
    """
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)


def conv_prelu(n=512, kernel_size=(2, 2), padding='same'):
    def inside(x):
        x = Conv2D(filters=n, kernel_size=kernel_size, padding=padding)(x)
        # BatchNorm
        x = PRELU()(x)
        return x
    return inside


def vh_concat(n=512, kernel_size=2, padding='same'):
    def inside(x):
        x1 = conv_prelu(n=n, kernel_size=(kernel_size, 1), padding=padding)(x)
        x2 = conv_prelu(n=n, kernel_size=(1, kernel_size), padding=padding)(x)
        x3 = Concatenate()([x1, x2])
        return x3

    return inside


def depthwise_conv_prelu(n=16, kernel_size=(2, 2), padding='same'):
    def inside(x):
        x = DepthwiseConv2D(depth_multiplier=n, kernel_size=kernel_size, padding=padding)(x)
        # BatchNorm
        x = PRELU()(x)
        return x
    return inside


def depthwise_vh_concat(n=16, kernel_size=2, padding='same'):
    def inside(x):
        x1 = depthwise_conv_prelu(n=n, kernel_size=(kernel_size, 1), padding=padding)(x)
        x2 = depthwise_conv_prelu(n=n, kernel_size=(1, kernel_size), padding=padding)(x)
        x3 = Concatenate()([x1, x2])
        return x3

    return inside



