# -*- coding:utf-8 -*-

from recommend_models.recommend_Model import gx_model
import numpy as np
from keras.layers.core import Dropout, Reshape
from keras.layers import Lambda, Concatenate, Add
from keras.layers import Dense, Input, AveragePooling2D, concatenate
from keras.models import Model
from keras import backend as K
from process_text.processing_data import process_data, get_mashup_api_allCategories
import tensorflow as tf
from embedding.encoding_padding_texts import encoding_padding

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size = 3


class gx_text_tag_model(gx_model):
    """
    同时处理text和tag的结构；新加入feature特征提取器;但不加入MF部分
    """

    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type='all', tag_manner='new_average', merge_manner='final_merge', text_fc_unit_nums=[],
                 tag_fc_unit_nums=[],
                 mf_embedding_dim=50, mf_fc_unit_nums=[]):
        """
        :param content_fc_unit_nums: 获取文本部分的feature，和mf部分平级，必须要有
        :param text_fc_unit_nums:只有当merge_manner='final_merge'时需要
        :param tag_fc_unit_nums:  同上
        """
        super(gx_text_tag_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                text_extracter_mode,
                                                inception_channels, inception_pooling, inception_fc_unit_nums,
                                                content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums)

        self.categories_feature_extracter = None  # 用于提取categories中的特征，和text公用一个embedding层

        self.Category_type = Category_type
        self.tag_manner = tag_manner
        self.merge_manner = merge_manner

        self.tag_fc_unit_nums = tag_fc_unit_nums
        self.text_fc_unit_nums = text_fc_unit_nums

    def get_name(self):
        name = super(gx_text_tag_model, self).get_name()
        name += 'text_fc_unit_nums:{} '.format(self.text_fc_unit_nums).replace(',', ' ');
        name += 'tag_fc_unit_nums:{} '.format(self.tag_fc_unit_nums).replace(',', ' ');
        name += 'Category_type:{} tag_manner:{} merge_manner:{}'.format(self.Category_type, self.tag_manner,
                                                                        self.merge_manner)

        return 'gx_text_tag_model:' + name  # ***

    def process_text(self):  # 处理文本，先进行
        """
        process mashup and service together
        :param data_dirs: 某个路径下的mashup和api文档
        :return:
        """
        mashup_descriptions, api_descriptions, mashup_categories, api_categories = self.pd.get_all_texts(
            self.Category_type)
        descriptions = mashup_descriptions + api_descriptions + mashup_categories + api_categories  # 先mashup后api 最后是类别
        """
        with open('../data/all_texts','w',encoding='utf-8') as f:
            for text in descriptions:
                f.write('{}\n'.format(text))
        """
        self.encoded_texts = encoding_padding(descriptions, self.remove_punctuation)  # 可得到各文本的encoded形式

    def get_categories_feature_extracter(self):
        """
        跟标签种类和得到向量的方式都有关系
        :return:
        """
        if self.tag_manner == 'old_average':
            return self.get_categories_old_feature_extracter()
        elif self.tag_manner == 'new_average':
            return self.get_categories_average_feature_extracter()
        elif self.tag_manner == 'merging':
            return self.get_categories_merging_feature_extracter()

    def get_categories_old_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；老写法  有点问题***
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_embedding_layer()
            embedded_results = embedding_layer(categories_input)  # 转化为2D
            embedded_sequences = Lambda(lambda x: tf.expand_dims(x, axis=3))(
                embedded_results)  # tf 和 keras的tensor 不同！！！
            # average sum/size size变量
            embedded_results = AveragePooling2D(pool_size=(MAX_SEQUENCE_LENGTH, 1))(
                embedded_sequences)  # 输出(None,1,embedding,1)?
            embedded_results = Reshape((self.embedding_dim,), name='categories_feature_extracter')(
                embedded_results)  # 为了能够跟text得到的 (None,1,embedding) merge

            self.categories_feature_extracter = Model(categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_average_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            # size=len(np.nonzero(categories_input)) # 词汇数 对tensor不能使用nonzero！
            embedding_layer = self.get_embedding_layer()
            embedded_results = embedding_layer(categories_input)  # 转化为2D (samples, sequence_length, output_dim)
            added = Add()(
                embedded_results)  # 沿 哪个轴？sequence_length？# 输出(samples, 1, output_dim)   必须是a list；且'Tensor' object is not iterable.不可*或list
            # embedded_results = Lambda(lambda x: tf.divide(added,size))(added)  # tf 和 keras的tensor 不同！！！可以将向量每个元素除以实数？
            self.categories_feature_extracter = Model(categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_merging_feature_extracter(self):
        """
        categories_feature的特征提取器    整合最多三个类别词的embedding
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_embedding_layer()
            embedded_results = embedding_layer(categories_input)

            getindicelayer = Lambda(lambda x: x[:, -1 * categories_size:, :])
            embedded_results = getindicelayer(embedded_results)  # 三个词的embedding # (samples, 3, output_dim)
            print('before mering, shape of sliced embedding :')
            print(embedded_results.shape)

            # results=concatenate([embedded_results[0],embedded_results[1],embedded_results[2]]) # 默认最后一个轴 # (samples, 1, 3*output_dim)
            # bug：AttributeError: 'Tensor' object has no attribute '_keras_history'  直接索引的tensor是tf中tensor

            getindicelayer1 = Lambda(lambda x: x[:, 0, :])
            layer1 = getindicelayer1(embedded_results)
            print('before mering,shape:')
            print(layer1.shape)

            getindicelayer2 = Lambda(lambda x: x[:, 1, :])
            getindicelayer3 = Lambda(lambda x: x[:, 2, :])
            results = Concatenate(name='categories_feature_extracter')(
                [layer1, getindicelayer2(embedded_results), getindicelayer3(embedded_results)])
            print('mering 3 embedding,shape:')
            print(results.shape)

            self.categories_feature_extracter = Model(categories_input, results)
        return self.categories_feature_extracter

    def get_text_tag_part(self, user_text_input, item_text_input, user_categories_input, item_categories_input):
        """
        同时处理text和tag
        :param user_text_input:
        :param item_text_input:
        :return:
        """
        user_text_feature = self.feature_extracter_from_texts()(user_text_input)  # (None,1,embedding)?
        item_text_feature = self.feature_extracter_from_texts()(item_text_input)

        user_categories_feature = self.get_categories_feature_extracter()(user_categories_input)
        item_categories_feature = self.get_categories_feature_extracter()(item_categories_input)

        if self.merge_manner == 'direct_merge':
            x = Concatenate()([user_text_feature, item_text_feature, user_categories_feature,
                               item_categories_feature])  # 整合文本和类别特征，尽管层次不太一样

        elif self.merge_manner == 'final_merge':

            x = concatenate([user_text_feature, item_text_feature])
            for unit_num in self.text_fc_unit_nums:
                x = Dense(unit_num, activation='relu')(x)

            y = concatenate([user_categories_feature, item_categories_feature])
            for unit_num in self.tag_fc_unit_nums:
                y = Dense(unit_num, activation='relu')(y)

            x = concatenate([x, y])  # 整合文本和类别特征，尽管层次不太一样

        for unit_num in self.content_fc_unit_nums[:-1]:
            x = Dense(unit_num, activation='relu')(x)

        x = Dense(self.content_fc_unit_nums[-1], activation='relu', name='text_tag_feature_extracter')(x)

        print('built textCNN layer, done!')
        return x

    def get_model(self):
        # right part
        user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

        user_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
        item_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

        x = self.get_text_tag_part(user_text_input, item_text_input, user_categories_input, item_categories_input)

        predict_vector = Dropout(0.5)(x)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)

        self.model = Model(inputs=[user_text_input, item_text_input, user_categories_input, item_categories_input],
                           outputs=[predict_result])

        print('built whole model, done!')
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances,mashup_only=False):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！
        :param args:
        :return:
        """
        pd = process_data(self.base_dir, False)
        mashup_id2info = pd.get_mashup_api_id2info('mashup')
        api_id2info = pd.get_mashup_api_id2info('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        if mashup_only:
            examples = (
                np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
                np.array(self.encoded_texts.get_texts_in_index(mashup_categories, 'self_padding')),
            )
        else:
            examples = (
                np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
                np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
                np.array(self.encoded_texts.get_texts_in_index(mashup_categories, 'self_padding')),
                np.array(self.encoded_texts.get_texts_in_index(api_categories, 'self_padding'))
            )

        return examples


class gx_text_tag_MF_model(gx_text_tag_model):
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type, tag_manner, merge_manner,
                 mf_embedding_dim, mf_fc_unit_nums,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[]):
        super(gx_text_tag_MF_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                   text_extracter_mode, inception_channels,
                                                   inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                                                   Category_type, tag_manner, merge_manner, text_fc_unit_nums,
                                                   tag_fc_unit_nums,
                                                   mf_embedding_dim, mf_fc_unit_nums)
        self.predict_fc_unit_nums = predict_fc_unit_nums  # 用于整合文本和mf之后的预测

    def get_name(self):
        name = super(gx_text_tag_MF_model, self).get_name()
        name += 'predict_fc_unit_nums:{} '.format(self.predict_fc_unit_nums).replace(',', ' ');
        return 'gx_text_tag_MF_model:' + name  # ***

    def get_model(self):
        # right part
        user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

        user_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
        item_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

        x = self.get_text_tag_part(user_text_input, item_text_input, user_categories_input, item_categories_input)
        # x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        # print(x.shape) # (?, 1, 50)

        # left part
        user_id = Input(shape=(1,), dtype='int32', name='user_input')  # 一个数字
        item_id = Input(shape=(1,), dtype='int32', name='item_input')
        # y = self.get_mf_part(user_id, item_id)

        # MLP part  使用接口model
        mf_mlp = self.get_mf_MLP(self.num_users, self.num_items, self.mf_embedding_dim, self.mf_fc_unit_nums)
        y = mf_mlp([user_id, item_id])
        # print(y.shape) # (?, 1, 50)

        # merge the two parts
        predict_vector = concatenate([x, y])
        print('final merge,done!')
        print(predict_vector.shape)  # (?, 1, 100)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense(unit_num, activation='relu')(predict_vector)

        # predict_vector = Flatten()(predict_vector)
        predict_vector = Dropout(0.5)(predict_vector)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)

        self.model = Model(
            inputs=[user_id, item_id, user_text_input, item_text_input, user_categories_input, item_categories_input],
            outputs=[predict_result])

        print('built whole model, done!')
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances):
        pd = process_data(self.base_dir, False)
        mashup_id2info = pd.get_mashup_api_id2info('mashup')
        api_id2info = pd.get_mashup_api_id2info('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id
                             in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        examples = (
            np.array(mashup_id_instances),
            np.array(api_id_instances),
            np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
            np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
            np.array(self.encoded_texts.get_texts_in_index(mashup_categories, 'self_padding')),
            np.array(self.encoded_texts.get_texts_in_index(api_categories, 'self_padding'))
        )

        return examples


class gx_text_tag_CF_model(gx_text_tag_model):
    # mf_fc_unit_nums 部分没有用
    # mashup_api_matrix 是U-I 调用矩阵；ini_features_array：mashup的text和tag训练好的整合的特征，初始化使用；max_ks 最大的topK个邻居
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type, tag_manner, merge_manner,
                 mf_embedding_dim, mf_fc_unit_nums,
                 mashup_api_matrix, ini_features_array, max_ks,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[]):

        super(gx_text_tag_CF_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                   text_extracter_mode, inception_channels,
                                                   inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                                                   Category_type, tag_manner, merge_manner, text_fc_unit_nums,
                                                   tag_fc_unit_nums,
                                                   mf_embedding_dim, mf_fc_unit_nums)

        self.mashup_api_matrix = mashup_api_matrix
        self.update_features(ini_features_array)
        self.max_ks = max_ks
        self.max_k = max_ks[-1]  # 一定要小于num——users！
        self.predict_fc_unit_nums = predict_fc_unit_nums  # 用于整合文本和mf之后的预测

    def update_features(self,ini_features_array):
        self.features = tf.Variable(ini_features_array, dtype='float32',
                                    trainable=False)  # 存储所有mashup特征的变量  tf中的tensor  需要在必要处区分keras的tensor
    def get_model(self, text_tag_model=None):

        user_text_input,item_text_input,user_categories_input,item_categories_input,x,mashup_feature=None,None,None,None,None,None

        if text_tag_model is None: # 根据结构从头开始搭建模型
            # 文本特征提取部分right part
            user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32',
                                    name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
            item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')
            user_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
            item_categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

            x = self.get_text_tag_part(user_text_input, item_text_input, user_categories_input,
                                       item_categories_input)  # 整合了mashup，api对的特征
            # print(x.shape)
            mashup_feature = concatenate([self.user_text_feature,
                                          self.user_categories_feature])  #有问题，还没改*** get_text_tag_part步骤中获取的mashup的text和tag feature
            #print(mashup_feature.shape)

        else: # 利用传入的搭建好的模型搭建模型
            user_text_input, item_text_input, user_categories_input, item_categories_input=text_tag_model.inputs

            x = text_tag_model.get_layer('text_tag_feature_extracter').output  # 获取的文本特征部分
            """
            text_feature=text_tag_model.get_layer('model_1').get_output_at(0)
            tag_feature = text_tag_model.get_layer('model_2').get_output_at(0)
            """
            # 针对的是direct merge的方法，整合用户的特征
            text_feature=text_tag_model.get_layer('concatenate_1').input[0]
            tag_feature = text_tag_model.get_layer('concatenate_1').input[2]
            mashup_feature = Concatenate()([text_feature,tag_feature])

        # CF part
        U_I = K.variable(self.mashup_api_matrix, dtype='int32')

        user_id = Input(shape=(1,), dtype='int32', name='user_input')  # 返回1D tensor,可当做变量的索引
        item_id = Input(shape=(1,), dtype='int32', name='item_input')

        def lam(paras):
            with tf.name_scope('topK_sim'):
                lam_user_id = paras[0]
                lam_item_id = paras[1]
                lam_mashup_feature = paras[2]
                lam_pair_feature = paras[3]

                # 每个样本的数据进行相同的处理
                def fn(elements):
                    # ***为什么搭建时是int32，使用Lambda层传入数据后自动变为float32?***
                    a_user_id = tf.squeeze(tf.cast(elements[0], tf.int32))  # scalar shape: (1,)  要作为索引要转化为()***
                    a_item_id = tf.squeeze(tf.cast(elements[1], tf.int32))  # scalar
                    a_mashup_feature = elements[2]  # 1D tensor
                    pair_feature = elements[3]

                    # tf.assign(self.features[a_user_id], a_mashup_feature)  # 何时更新？？？ 只有batch更新才有用

                    indexes = []
                    sims = []
                    same_sim_scalar=tf.constant(1.0)
                    small_tr=tf.constant(0.00001)
                    same_scalar = tf.constant(0.0)
                    for index in range(self.features.shape[0]): # 跟sim有关索引的全部是局部index
                        indexes.append(index)
                        sim=tensor_sim(a_mashup_feature, self.features[index])
                        final_sim = tf.cond(tf.abs(sim-same_sim_scalar)<=small_tr,lambda:same_scalar, lambda: sim) # 如果输入的feature和历史近似（float近似），认为相等，设为0
                        sims.append(final_sim)  # list of scalar

                    topK_prod = []
                    tensor_sims = [K.expand_dims(sim) for sim in sims]
                    tensor_sims = K.concatenate(tensor_sims)  # shape=(n,)
                    # print(tensor_sims.shape)

                    max_indexes = tf.nn.top_k(tensor_sims, self.max_k)[1]
                    for i in range(self.max_k):
                        index = max_indexes[i]
                        temp_sim = tensor_sims[index]
                        u_i = U_I[index][a_item_id]
                        topK_prod.append(temp_sim * tf.cast(u_i, tf.float32))

                    topk_sim_features = [K.expand_dims(sum(topK_prod[:topK])) for topK in
                                         self.max_ks]  # 各个topK下计算的sim积  tensor
                    CF_feature = K.concatenate(topk_sim_features)  # 整合的tensor 形状？
                    final_feature = tf.concat([CF_feature, pair_feature], 0)
                    return a_user_id, a_item_id, final_feature, pair_feature  # 同时返回user_id是为了保证输入和输出形状相同，user_id无实质意义

                _1, _2, final_feature, _3 = K.map_fn(fn,
                                                     (lam_user_id, lam_item_id, lam_mashup_feature, lam_pair_feature))
                return final_feature

        predict_vector = Lambda(lam)([user_id, item_id, mashup_feature, x])
        print(predict_vector.shape)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense(unit_num, activation='relu')(predict_vector)

        # predict_vector = Flatten()(predict_vector)
        predict_vector = Dropout(0.5)(predict_vector)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)

        self.model = Model(
            inputs=[user_id, item_id, user_text_input, item_text_input, user_categories_input, item_categories_input],
            outputs=[predict_result])

        print('built whole model, done!')
        return self.model

    def get_name(self):
        name = super(gx_text_tag_CF_model, self).get_name()
        name += 'max_ks:{} '.format(self.max_ks).replace(',', ' ');
        name += 'predict_fc_unit_nums:{} '.format(self.predict_fc_unit_nums).replace(',', ' ');
        return 'gx_text_tag_CF_model:' + name  # ***

    def get_instances(self, mashup_id_instances, api_id_instances):
        pd = process_data(self.base_dir, False)
        mashup_id2info = pd.get_mashup_api_id2info('mashup')
        api_id2info = pd.get_mashup_api_id2info('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id
                             in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        # 针对使用预训练的text_tag_model(embedding复用）
        if self.encoded_texts is None:
            self.process_text()
        # print('train examples:' + str(len(mashup_id_instances)))
        examples = (
            np.array(mashup_id_instances),
            np.array(api_id_instances),
            np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
            np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
            np.array(self.encoded_texts.get_texts_in_index(mashup_categories, 'self_padding')),
            np.array(self.encoded_texts.get_texts_in_index(api_categories, 'self_padding'))
        )

        """
        if api_id_instances[0]!=api_id_instances[-1]: # 不保存feature用例的
            np.savetxt('../data/getInstences_encoding_texts1', examples[2],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts2', examples[3],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts3', examples[4],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts4', examples[5],fmt='%d')
            print('save getInstences_encoding_texts,done!')
        """
        return examples


# 搭建模型阶段 抽象tensor的运算
def tensor_sim(f1, f2):
    fenmu = K.sum(tf.multiply(f1, f2))
    sum1 = K.sqrt(K.sum(K.square(f1)))
    sum2 = K.sqrt(K.sum(K.square(f2)))
    return fenmu / (sum1 * sum2)
