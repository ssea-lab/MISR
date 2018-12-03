# -*- coding:utf-8 -*-
import os

from keras.layers import Dense, Input, Dropout, Lambda, Concatenate
import numpy as np
from keras import backend as K, Model
import tensorflow as tf


# 构建阶段应当使用抽象化的变量和占位符，而不是array；也不能用sess查询值？

#  改写排序：如果 x 应该排在 y 的前面，返回 -1，如果 x 应该排在 y 的后面，返回 1。如果 x 和 y 相等，返回 0。
def _cmp(x, y):
    def f1():
        return -1

    def f2():
        return 1

    return tf.cond(K.greater(x[1], y[1]), f1, f2) # cond返回tensor  这里是 shape=() dtype=int32

    """
    # 需要根据tensor的取值采取不同的操作
    if K.greater(x[1],y[1]) is not None:
        return -1
    if K.less(x[1],y[1]) is not None:
        return 1
    return 0    
    """


ini_features_array= np.zeros((3,8))
# 存储所有mashup特征的变量  tf中的tensor  需要在必要处区分keras的tensor
features = tf.Variable(ini_features_array,dtype='float32',trainable=False)
max_ks = [1,2]#[5, 10, 20, 30, 40, 50]
max_k = max_ks[-1]


def get_model_test(num_users = 3, mashup_api_matrix=np.random.random((3,3))):
    MAX_SEQUENCE_LENGTH = 150

    # right part
    user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32',
                            name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量

    x = Dense(8, activation='relu')(user_text_input)  # 8D  tensor
    # print(x.shape)

    # x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
    # print(x.shape) # (?, 1, 50)

    # CF part
    U_I= K.variable(mashup_api_matrix,dtype='int32')

    user_id = Input(shape=(1,), dtype='int32', name='user_input')  # 返回1D tensor,可当做变量的索引
    item_id = Input(shape=(1,), dtype='int32', name='item_input')

    def fn(elements):
        a_user_id=tf.squeeze(elements[0]) # scalar shape:(1,)  要作为索引要转化为()
        a_item_id=tf.squeeze(elements[1]) # scalar
        a_mashup_feature=elements[2] # 1D tensor

        tf.assign(features[a_user_id], a_mashup_feature)  # 何时更新？？？

        indexes = []
        sims = []
        for index in range(num_users):
            indexes.append(index) # 包括自身，不易判断值再剔除
            sims.append(tensor_sim(a_mashup_feature, features[index])) # list of scalar

        """
        sim_mapping = [pair for pair in zip(indexes, sims)]
        k_pairs = sorted(sim_mapping, key=cmp_to_key(_cmp))  # 比较的是一维tensor的值,搭建网络时值不确定，没法直接比较，需要用cond函数
        # cond返回的是嵌套+-1值的tensor，无法直接应用到比较器中，无法解决这一问题***
        
        max_k_pairs = k_pairs[:max_k+1]

        # scalar 相乘，仍是scalar
        topK_prod = [sim_tensor * U_I[mashup_index][a_item_id] for mashup_index, sim_tensor in max_k_pairs[1:]]  # 最近的某个mashup的sim*要归一化？ 调用该api的0-1
        """

        topK_prod=[]
        tensor_sims=[K.expand_dims(sim) for sim in sims]
        tensor_sims=K.concatenate(tensor_sims) # shape=(n,)
        print(tensor_sims.shape)

        max_indexes = tf.nn.top_k(tensor_sims, max_k+1)[1]
        for i in range(1,max_k+1):
            index= max_indexes[i]
            temp_sim=tensor_sims[index]
            u_i=U_I[index][a_item_id]
            topK_prod.append(temp_sim*tf.cast(u_i,tf.float32))

        topk_sim_features = [K.expand_dims(sum(topK_prod[:topK])) for topK in max_ks]  # 各个topK下计算的sim积  tensor
        CF_feature = K.concatenate(topk_sim_features)  # 整合的tensor 形状？
        return a_user_id,a_item_id,CF_feature # 同时返回user_id是为了保证输入和输出形状相同，user_id无实质意义

    _1,_2,CF_features=K.map_fn(fn,(user_id,item_id,x))

    # merge the two parts
    """
    不能这么操作,从map_fn方法种获取的CF_features已经是tf.tensor，没法转化为keras的tensor
    需要使用一个大的lambda层：输入keras.tensor；lambda函数内部可以对输入的keras.tensor进行各种tf的操作 ，如get_model()
    """
    def lam(tensors):
        output=tf.concat([tensors[0], tensors[1]],1)
        return output

    predict_vector = Lambda(lam)([x, CF_features])

    print('final merge,done!')
    print(predict_vector.shape)  # (?, 1, 100)

    # predict_vector = Flatten()(predict_vector)
    predict_vector = Dropout(0.5)(predict_vector)
    predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
        predict_vector)

    model = Model(
        inputs=[user_id, item_id, user_text_input],
        outputs=[predict_result])

    print('built whole model, done!')
    return model


def get_model(num_users=3, mashup_api_matrix=np.random.random((3,3))):
    global features,max_ks,max_k
    MAX_SEQUENCE_LENGTH = 150

    # right part
    user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32',
                            name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量

    x = Dense(8, activation='relu')(user_text_input)  # 8D  tensor

    # CF part
    U_I = K.variable(mashup_api_matrix, dtype='int32')

    user_id = Input(shape=(1,), dtype='int32', name='user_input')  # 返回1D tensor,可当做变量的索引
    item_id = Input(shape=(1,), dtype='int32', name='item_input')

    def lam(paras):
        with tf.name_scope('Channel_Simple'):
            lam_user_id=paras[0]
            lam_item_id=paras[1]
            lam_x=paras[2]

            # 每个样本的数据进行相同的处理
            def fn(elements):
                # ***为什么搭建时是int32，使用Lambda层传入数据后自动变为float32?***
                a_user_id = tf.squeeze(tf.cast(elements[0],tf.int32))  # scalar shape:(1,)  要作为索引要转化为()
                a_item_id = tf.squeeze(tf.cast(elements[1],tf.int32)) # scalar
                a_mashup_feature = elements[2]  # 1D tensor

                tf.assign(features[a_user_id], a_mashup_feature)  # 何时更新？？？

                indexes = []
                sims = []
                for index in range(num_users):
                    indexes.append(index)  # 包括自身，不易判断值再剔除
                    sims.append(tensor_sim(a_mashup_feature, features[index]))  # list of scalar

                topK_prod = []
                tensor_sims = [K.expand_dims(sim) for sim in sims]
                tensor_sims = K.concatenate(tensor_sims)  # shape=(n,)
                # print(tensor_sims.shape)

                max_indexes = tf.nn.top_k(tensor_sims, max_k + 1)[1]
                for i in range(1, max_k + 1):
                    index = max_indexes[i]
                    temp_sim = tensor_sims[index]
                    u_i = U_I[index][a_item_id]
                    topK_prod.append(temp_sim * tf.cast(u_i, tf.float32))

                topk_sim_features = [K.expand_dims(sum(topK_prod[:topK])) for topK in max_ks]  # 各个topK下计算的sim积  tensor
                CF_feature = K.concatenate(topk_sim_features)  # 整合的tensor 形状？
                final_feature =tf.concat([CF_feature,a_mashup_feature],0)
                return a_user_id, a_item_id, final_feature  # 同时返回user_id是为了保证输入和输出形状相同，user_id无实质意义

            _1, _2, final_feature = K.map_fn(fn, (lam_user_id, lam_item_id, lam_x))
            return final_feature

    predict_vector=Lambda(lam)([user_id,item_id, x])
    print(predict_vector.shape)

    predict_vector=Dense(10,activation='relu')(predict_vector)
    print('final merge,done!')

    # predict_vector = Flatten()(predict_vector)
    predict_vector = Dropout(0.5)(predict_vector)
    predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
        predict_vector)

    model = Model(
        inputs=[user_id, item_id, user_text_input],
        outputs=[predict_result])

    print('built whole model, done!')
    return model

# 搭建模型阶段 抽象tensor的运算
def tensor_sim(f1, f2):
    fenmu = K.sum(tf.multiply(f1, f2))
    sum1 = K.sqrt(K.sum(K.square(f1)))
    sum2 = K.sqrt(K.sum(K.square(f2)))
    return fenmu / (sum1 * sum2)

def get_middle_output():
    inputs = Input(shape=(784,))
    inputs1 = Input(shape=(784,))
    con=Concatenate()([inputs,inputs1])
    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(con)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax',name='final_dense1')(x)
    model1= Model(inputs=[inputs,inputs1], outputs=predictions)
    for layer in model1.layers:
        print(layer.name)
    print('model1 done!')

    y = Dense(10, activation='softmax', name='final_dense2')(model1.output)
    # This creates a model that includes
    # the Input layer and three Dense layers
    model2 = Model(inputs=model1.input, outputs=y) #model1.get_layer('final_dense1').output
    print('model2 done!')
    model3= Model(inputs=model2.input, outputs=model2.get_layer('final_dense1').output)
    for layer in model3.layers:
        print(layer.name)
    print('model3 done!')
    return 0

if __name__ == '__main__':
    # model = get_model()
    # get_middle_output()
    print(os.path.abspath('.'))
    print(os.path.abspath('..'))