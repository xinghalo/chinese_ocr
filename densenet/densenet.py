from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed

"""
Densenet的特点：
1 特征复用
2 
"""
def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    # 输出的维度是8
    x = Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same')(x)
    if(dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    # growth_rate是增长率 = 8，表示的是卷积核的数量
    # 把输入与每一个卷积层做了一个拼接
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    # 再次做了一次卷积
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if(dropout_rate):
        x = Dropout(dropout_rate)(x)

    # 根据池化的类型，选择池化方法
    # 2 代表不重复的平均池化
    # 1 代表边界填充0，然后覆盖率50%的平均池化
    # 3 代表覆盖率50%的平均池化
    if(pooltype == 2):
        # 通过(2,2)的卷积，使得参数为原来的一半
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif(pooltype == 1):
        x = ZeroPadding2D(padding = (0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif(pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter

def dense_cnn(input, nclass):
    # input是多少维的？都是(32,x,1)
    _dropout_rate = 0.2 
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    # TODO 针对卷积还是不熟！
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
   
    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    # todo permute和timedistributed怎么理解？
    x = Permute((2, 1, 3), name='permute')(x)
    # 这里基于每个时间序列，都有一个softmax输出
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()
    # todo 这个pred是什么？
    return y_pred

def dense_blstm(input):

    pass

input = Input(shape=(32, 280, 1), name='the_input')
dense_cnn(input, 5000)
