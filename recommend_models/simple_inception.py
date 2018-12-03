# -*- coding:utf-8 -*-

from keras import backend as K
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, Embedding, Activation, merge, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.applications import inception_v3

channel_axis = 1 if K.image_data_format() == 'channels_first' else 3


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`for the convolution and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters, (num_row, num_col), strides=strides,
               padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_layer(x, EMBEDDING_DIM, filterNums=(64,96,64,96,64),pooling='max'):
    """
    使用CNN提取文本特征时可以使用的inception结构，本质上是输入x输出x
    filterNums为各种kernel的数目  0 1 3 4 之和为最终feature map个数，global max时为特征维度
    eg:1*300   3*300，  替代5*300的3*300,3*1    pool 3*1，1*300
    一个类似的经典数目组合是64 96,  64,96，  64
    :param x:
    :param filterNums:
    :param EMBEDDING_DIM: word embedding维度？
    :return:
    """

    branch1x300 = conv2d_bn(x, filterNums[0], 1, EMBEDDING_DIM)

    branch3x300 = conv2d_bn(x, filterNums[1], 3, EMBEDDING_DIM)

    # 5*EMBEDDING_DIM 拆分为3*EMBEDDING_DIM  3*1
    branch5x300dbl = conv2d_bn(x, filterNums[2], 3, EMBEDDING_DIM)
    branch5x300dbl = conv2d_bn(branch5x300dbl, filterNums[3], 3, 1)

    # 3*EMBEDDING_DIM  first pooling by (3, 1), then conv by 1*300
    branch_pool = AveragePooling2D((3, 1), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, filterNums[4], 1, EMBEDDING_DIM)

    x = concatenate(
        [branch1x300, branch3x300, branch5x300dbl, branch_pool],
        axis=channel_axis
        , name='mixed1')#

    if pooling == 'global_avg':
        x = GlobalAveragePooling2D()(x) # 一维  维度等于feature map 数目
    elif pooling == 'global_max':
        x = GlobalMaxPooling2D()(x)
    elif pooling == 'max': # 传统池化+FC 提取特征
        x = MaxPooling2D()(x) # filter num* len
        x = Flatten()(x) # 1D

    return x
