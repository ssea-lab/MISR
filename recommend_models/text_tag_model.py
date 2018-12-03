# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from helpers.util import cos_sim
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
channel_axis = 1 if K.image_data_format () == 'channels_first' else 3
categories_size = 3


class gx_text_tag_model (gx_model):
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
        super (gx_text_tag_model, self).__init__ (base_dir, remove_punctuation, embedding_name, embedding_dim,
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
        name = super (gx_text_tag_model, self).get_name ()
        name += 'text_fc_unit_nums:{} '.format (self.text_fc_unit_nums).replace (',', ' ');
        name += 'tag_fc_unit_nums:{} '.format (self.tag_fc_unit_nums).replace (',', ' ');
        name += 'Category_type:{} tag_manner:{} merge_manner:{}'.format (self.Category_type, self.tag_manner,
                                                                         self.merge_manner)

        return 'gx_text_tag_model:' + name  # ***

    def process_text(self):  # 处理文本，先进行
        """
        process mashup and service together
        :param data_dirs: 某个路径下的mashup和api文档
        :return:
        """
        mashup_descriptions, api_descriptions, mashup_categories, api_categories = self.pd.get_all_texts (
            self.Category_type)
        descriptions = mashup_descriptions + api_descriptions + mashup_categories + api_categories  # 先mashup后api 最后是类别
        """
        with open('../data/all_texts','w',encoding='utf-8') as f:
            for text in descriptions:
                f.write('{}\n'.format(text))
        """
        self.encoded_texts = encoding_padding (descriptions, self.remove_punctuation)  # 可得到各文本的encoded形式

    def get_categories_feature_extracter(self):
        """
        跟标签种类和得到向量的方式都有关系
        :return:
        """
        if self.tag_manner == 'old_average':
            return self.get_categories_old_feature_extracter ()
        elif self.tag_manner == 'new_average':
            return self.get_categories_average_feature_extracter ()
        elif self.tag_manner == 'merging':
            return self.get_categories_merging_feature_extracter ()

    def get_categories_old_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；老写法  有点问题***
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_embedding_layer ()
            embedded_results = embedding_layer (categories_input)  # 转化为2D
            embedded_sequences = Lambda (lambda x: tf.expand_dims (x, axis=3)) (
                embedded_results)  # tf 和 keras的tensor 不同！！！
            # average sum/size size变量
            embedded_results = AveragePooling2D (pool_size=(MAX_SEQUENCE_LENGTH, 1)) (
                embedded_sequences)  # 输出(None,1,embedding,1)?
            embedded_results = Reshape ((self.embedding_dim,), name='categories_feature_extracter') (
                embedded_results)  # 为了能够跟text得到的 (None,1,embedding) merge

            self.categories_feature_extracter = Model (categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_average_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            # size=len(np.nonzero(categories_input)) # 词汇数 对tensor不能使用nonzero！
            embedding_layer = self.get_embedding_layer ()
            embedded_results = embedding_layer (categories_input)  # 转化为2D (samples, sequence_length, output_dim)
            added = Add () (
                embedded_results)  # 沿 哪个轴？sequence_length？# 输出(samples, 1, output_dim)   必须是a list；且'Tensor' object is not iterable.不可*或list
            # embedded_results = Lambda(lambda x: tf.divide(added,size))(added)  # tf 和 keras的tensor 不同！！！可以将向量每个元素除以实数？
            self.categories_feature_extracter = Model (categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_merging_feature_extracter(self):
        """
        categories_feature的特征提取器    整合最多三个类别词的embedding
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_embedding_layer ()
            embedded_results = embedding_layer (categories_input)

            getindicelayer = Lambda (lambda x: x[:, -1 * categories_size:, :])
            embedded_results = getindicelayer (embedded_results)  # 三个词的embedding # (samples, 3, output_dim)
            print ('before mering, shape of sliced embedding :')
            print (embedded_results.shape)

            # results=concatenate([embedded_results[0],embedded_results[1],embedded_results[2]]) # 默认最后一个轴 # (samples, 1, 3*output_dim)
            # bug：AttributeError: 'Tensor' object has no attribute '_keras_history'  直接索引的tensor是tf中tensor

            getindicelayer1 = Lambda (lambda x: x[:, 0, :])
            layer1 = getindicelayer1 (embedded_results)
            print ('before mering,shape:')
            print (layer1.shape)

            getindicelayer2 = Lambda (lambda x: x[:, 1, :])
            getindicelayer3 = Lambda (lambda x: x[:, 2, :])
            results = Concatenate (name='categories_feature_extracter') (
                [layer1, getindicelayer2 (embedded_results), getindicelayer3 (embedded_results)])
            print ('mering 3 embedding,shape:')
            print (results.shape)

            self.categories_feature_extracter = Model (categories_input, results)
        return self.categories_feature_extracter

    def get_text_tag_part(self, user_text_input, item_text_input, user_categories_input, item_categories_input):
        """
        同时处理text和tag
        :param user_text_input:
        :param item_text_input:
        :return:
        """
        user_text_feature = self.feature_extracter_from_texts () (user_text_input)  # (None,1,embedding)?
        item_text_feature = self.feature_extracter_from_texts () (item_text_input)

        user_categories_feature = self.get_categories_feature_extracter () (user_categories_input)
        item_categories_feature = self.get_categories_feature_extracter () (item_categories_input)

        if self.merge_manner == 'direct_merge':
            x = Concatenate (name='concatenate_1') ([user_text_feature, item_text_feature, user_categories_feature,
                                 item_categories_feature])  # 整合文本和类别特征，尽管层次不太一样

        elif self.merge_manner == 'final_merge':

            x = concatenate ([user_text_feature, item_text_feature])
            for unit_num in self.text_fc_unit_nums:
                x = Dense (unit_num, activation='relu') (x)

            y = concatenate ([user_categories_feature, item_categories_feature])
            for unit_num in self.tag_fc_unit_nums:
                y = Dense (unit_num, activation='relu') (y)

            x = concatenate ([x, y])  # 整合文本和类别特征，尽管层次不太一样

        for unit_num in self.content_fc_unit_nums[:-1]:
            x = Dense (unit_num, activation='relu') (x)

        x = Dense (self.content_fc_unit_nums[-1], activation='relu', name='text_tag_feature_extracter') (x)

        print ('built textCNN layer, done!')
        return x

    def get_model(self):
        # right part
        user_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                 name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

        user_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
        item_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

        x = self.get_text_tag_part (user_text_input, item_text_input, user_categories_input, item_categories_input)

        predict_vector = Dropout (0.5) (x)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        self.model = Model (inputs=[user_text_input, item_text_input, user_categories_input, item_categories_input],
                            outputs=[predict_result])

        print ('built whole model, done!')
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances, mashup_only=False):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！
        :param args:
        :return:
        """

        mashup_id2info = self.pd.get_mashup_api_id2info ('mashup')
        api_id2info = self.pd.get_mashup_api_id2info ('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories ('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories ('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        if mashup_only:
            examples = (
                np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
            )
        else:
            examples = (
                np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
                np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
                np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding'))
            )

        return examples


class gx_text_tag_MF_model (gx_text_tag_model):
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type, tag_manner, merge_manner,
                 mf_embedding_dim, mf_fc_unit_nums,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[]):
        super (gx_text_tag_MF_model, self).__init__ (base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                     text_extracter_mode, inception_channels,
                                                     inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                                                     Category_type, tag_manner, merge_manner, text_fc_unit_nums,
                                                     tag_fc_unit_nums,
                                                     mf_embedding_dim, mf_fc_unit_nums)
        self.predict_fc_unit_nums = predict_fc_unit_nums  # 用于整合文本和mf之后的预测

    def get_name(self):
        name = super (gx_text_tag_MF_model, self).get_name ()
        name += 'predict_fc_unit_nums:{} '.format (self.predict_fc_unit_nums).replace (',', ' ');
        return 'gx_text_tag_MF_model:' + name  # ***

    def get_model(self):
        # right part
        user_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                 name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

        user_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
        item_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

        x = self.get_text_tag_part (user_text_input, item_text_input, user_categories_input, item_categories_input)
        # x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        # print(x.shape) # (?, 1, 50)

        # left part
        user_id = Input (shape=(1,), dtype='int32', name='user_input')  # 一个数字
        item_id = Input (shape=(1,), dtype='int32', name='item_input')
        # y = self.get_mf_part(user_id, item_id)

        # MLP part  使用接口model
        mf_mlp = self.get_mf_MLP (self.num_users, self.num_items, self.mf_embedding_dim, self.mf_fc_unit_nums)
        y = mf_mlp ([user_id, item_id])
        # print(y.shape) # (?, 1, 50)

        # merge the two parts
        predict_vector = concatenate ([x, y])
        print ('final merge,done!')
        print (predict_vector.shape)  # (?, 1, 100)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu') (predict_vector)

        # predict_vector = Flatten()(predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        self.model = Model (
            inputs=[user_id, item_id, user_text_input, item_text_input, user_categories_input, item_categories_input],
            outputs=[predict_result])

        print ('built whole model, done!')
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances):

        mashup_id2info = self.pd.get_mashup_api_id2info ('mashup')
        api_id2info = self.pd.get_mashup_api_id2info ('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories ('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id
                             in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories ('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        examples = (
            np.array (mashup_id_instances),
            np.array (api_id_instances),
            np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
            np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
            np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
            np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding'))
        )

        return examples


# item id 部分还要改动！！！
class gx_text_tag_CF_model (gx_text_tag_model):
    # mf_fc_unit_nums 部分没有用
    # mashup_api_matrix 是U-I 调用矩阵；ini_features_array：mashup的text和tag训练好的整合的特征，初始化使用；max_ks 最大的topK个邻居
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type, tag_manner, merge_manner,
                 mf_embedding_dim, mf_fc_unit_nums,
                 pmf_01, mashup_api_matrix, u_factors_matrix, i_factors_matrix, i_id_list, ini_features_array, max_ks,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[]):

        super (gx_text_tag_CF_model, self).__init__ (base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                     text_extracter_mode, inception_channels,
                                                     inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                                                     Category_type, tag_manner, merge_manner, text_fc_unit_nums,
                                                     tag_fc_unit_nums,
                                                     mf_embedding_dim, mf_fc_unit_nums)

        self.mashup_api_matrix = mashup_api_matrix
        self.u_factors_matrix = u_factors_matrix
        self.i_factors_matrix = i_factors_matrix
        self.i_id_list= tf.constant(i_id_list) # (n,)?
        self.pmf_01 = pmf_01

        self.max_ks = max_ks
        self.max_k = max_ks[-1]  # 一定要小于num——users！

        self.predict_fc_unit_nums = predict_fc_unit_nums  # 用于整合文本和mf之后的预测
        # 在一个batch的计算过程中，如果已经计算过某个mashup最近似的mashup，记录其局部index和sim，下个样本不需要计算；加速。不规范
        # 训练时每个batch之后若feature变化，则需要外部初始化。predict不需要更新

        self.feature_dim=len(ini_features_array[0])
        # self.id2SimMap={} 编码困难，用两个list代替
        self.update_features (ini_features_array)

    # embedding等文本部分参数可变时才有意义，batch更新后进行
    def update_features(self, ini_features_array):
        self.features = tf.Variable (ini_features_array, dtype='float32',
                                     trainable=False)  # 存储所有mashup特征的变量  tf中的tensor  需要在必要处区分keras的tensor
        # self.id2SimMap = {}
        self._userIds = []
        self.con_user_ids = None
        self._topkIndexes = []
        self.stack_topkIndexes = None
        self._topkSims = []
        self.stack_topkSims = None

    def get_model(self, text_tag_model=None):

        user_text_input, item_text_input, user_categories_input, item_categories_input, x, mashup_feature = None, None, None, None, None, None

        if text_tag_model is None:  # 根据结构从头开始搭建模型
            # 文本特征提取部分right part
            user_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='float32',
                                     name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
            item_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')
            user_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
            item_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

            x = self.get_text_tag_part (user_text_input, item_text_input, user_categories_input,
                                        item_categories_input)  # 整合了mashup，api对的特征
            # print(x.shape)
            mashup_feature = concatenate ([self.user_text_feature,
                                           self.user_categories_feature])  # 有问题，还没改*** get_text_tag_part步骤中获取的mashup的text和tag feature
            # print(mashup_feature.shape)

        else:  # 利用传入的搭建好的模型搭建模型
            user_text_input, item_text_input, user_categories_input, item_categories_input = text_tag_model.inputs

            x = text_tag_model.get_layer ('text_tag_feature_extracter').output  # 获取的文本特征部分
            """
            text_feature=text_tag_model.get_layer('model_1').get_output_at(0)
            tag_feature = text_tag_model.get_layer('model_2').get_output_at(0)
            """
            # 针对的是direct merge的方法，整合用户的特征
            text_feature = text_tag_model.get_layer ('concatenate_1').input[0]
            tag_feature = text_tag_model.get_layer ('concatenate_1').input[2]
            mashup_feature = Concatenate () ([text_feature, tag_feature])

        U_I, u_factors, i_factors = None, None, None
        # CF part
        if self.pmf_01 == '01':
            U_I = K.variable (self.mashup_api_matrix, dtype='int32')
        elif self.pmf_01 == 'pmf':
            u_factors = K.variable (self.u_factors_matrix, dtype='float32')  # 外部调用包，传入  存储
            i_factors = K.variable (self.i_factors_matrix, dtype='float32')

        user_id = Input (shape=(1,), dtype='int32', name='user_input')  # 返回1D tensor,可当做变量的索引
        item_id = Input (shape=(1,), dtype='int32', name='item_input')

        def lam(paras):
            with tf.name_scope ('topK_sim'):
                lam_user_id = paras[0]
                lam_item_id = paras[1]
                lam_mashup_feature = paras[2]

                # 每个样本的数据进行相同的处理
                def fn(elements):
                    # ***为什么搭建时是int32，使用Lambda层传入数据后自动变为float32?***
                    a_user_id = tf.cast (elements[0], tf.int32)  # 用来判断该mashup是否计算过最近邻
                    # 用来获取api的latent factor
                    # a_item_id = tf.squeeze (tf.cast (elements[1], tf.int32))  # scalar # scalar shape: (1,)  要作为索引要转化为()***
                    # 用来判断api是否存在于训练集中
                    a_item_id =tf.cast (elements[1], tf.int32)

                    a_mashup_feature = elements[2]  # 1D tensor

                    def cpt_top_sims():

                        sims = []
                        same_sim_scalar = tf.constant (1.0)
                        small_tr = tf.constant (0.00001)
                        same_scalar = tf.constant (0.0)
                        for index in range (self.features.shape[0]):  # 跟sim有关索引的全部是局部index
                            sim = tensor_sim (a_mashup_feature, self.features[index])
                            final_sim = tf.cond (tf.abs (sim - same_sim_scalar) <= small_tr, lambda: same_scalar,
                                                 lambda: sim)  # 如果输入的feature和历史近似（float近似），认为相等，设为0
                            sims.append (final_sim)  # list of scalar

                        tensor_sims = [K.expand_dims (sim) for sim in sims]
                        tensor_sims = K.concatenate (tensor_sims)  # shape=(n,)
                        # print(tensor_sims.shape)

                        max_sims, max_indexes = tf.nn.top_k (tensor_sims, self.max_k)  # shape=(n,)
                        max_sims = tf.squeeze (max_sims / tf.reduce_sum (max_sims))  # 归一化*** (1,n)->(n,)
                        # self.id2SimMap[a_user_id]=(max_sims,max_indexes)  # scalar(1,) -> (shape=(n,),shape=(n,))

                        self._userIds.append (a_user_id)  # scalar(1,)!!!
                        self.con_user_ids = K.concatenate (self._userIds)  # (n,)
                        self._topkIndexes.append (max_indexes)  # scalar(n,)
                        self._topkSims.append (max_sims)
                        self.stack_topkIndexes = tf.stack (self._topkIndexes)
                        self.stack_topkSims = tf.stack (self._topkSims)
                        print ('this mahsup has never been cpted!')

                        print ('max_sims and max_indexes in cpt_top_sims,')
                        print (max_sims)
                        print (max_indexes)
                        return [max_sims, max_indexes]

                    """
                    def get_top_sims():#
                        temp_returns=K.constant([0]*self.max_k,dtype='int32'),K.constant(np.zeros((self.max_k,)))
                        for temp_user_id in list(self.id2SimMap.keys()):
                            index_sim=tf.cond(tf.equal(temp_user_id, a_user_id),lambda:self.id2SimMap.get(temp_user_id),lambda:temp_returns)
                            if index_sim is not temp_returns:
                                return index_sim
                    """

                    def get_top_sims():  #
                        max_sims = tf.Variable (np.zeros ((self.max_k,)), dtype='float32')
                        max_indexes = tf.Variable (np.zeros ((self.max_k,)), dtype='int32')
                        if len (self._userIds) == 0:  # 不可能
                            return [max_sims, max_indexes]
                        else:
                            index = tf.constant (0)
                            if len (self._userIds) > 1:
                                index = tf.squeeze (
                                    tf.where (tf.equal (self.con_user_ids, a_user_id))[0])  # where:[?,1] -> [1]->()

                            max_sims = self.stack_topkSims[index]  # concatenate 对m个(n,)结果是（m*n,)；而stack是(m,n) [index]后是(n,)
                            max_indexes = self.stack_topkIndexes[index]
                        print ('this mahsup has been cpted!')
                        print ('max_sims and max_indexes in get_top_sims,')
                        print (max_sims)
                        print (max_indexes)
                        return [max_sims, max_indexes]

                    def scalar_in():
                        false = tf.constant (False, dtype='bool')
                        true = tf.constant (True, dtype='bool')
                        if len (self._userIds) == 0:
                            return false
                        elif len (self._userIds) == 1:
                            return tf.squeeze (tf.equal (self._userIds[0], a_user_id))
                        else:
                            # user_ids=K.concatenate(list(K.expand_dims(self.id2SimMap.keys()))) # shape=(n,)
                            is_in = tf.reduce_any (tf.equal (a_user_id, self.con_user_ids))
                            return is_in  # shape=() dtype=bool>

                    # *** scalar用in? 虽然值相同，但是对象不同.使用tf.equal
                    # lambda作为参数？函数callable？
                    max_sims, max_indexes = tf.cond (scalar_in (), get_top_sims, cpt_top_sims)

                    # 判断api id是否存在于训练集中
                    def i_id_In():
                        return tf.reduce_any (tf.equal (a_item_id, self.i_id_list))
                    i_local_index=tf.squeeze (tf.where (tf.equal (self.i_id_list, a_item_id))[0])

                    CF_feature = None
                    if self.pmf_01 == 'pmf':
                        sum_u_factor = K.variable (np.zeros_like (self.u_factors_matrix[0]))
                        for i in range (self.max_k):
                            index = max_indexes[i]
                            temp_sim = max_sims[i]
                            sum_u_factor += temp_sim * u_factors[index]

                        # 获取api的factor  id->局部索引
                        api_factor = tf.cond (i_id_In (),
                                              lambda: i_factors[i_local_index],
                                              tf.Variable (np.zeros((self.feature_dim,)), dtype='float32'))
                        # CF_feature = K.concatenate ([sum_u_factor, i_factors[a_item_id]]) 没注意id映射的错误写法
                        CF_feature = K.concatenate ([sum_u_factor, api_factor]) #

                    elif self.pmf_01 == '01':
                        topK_prod = []

                        for i in range (self.max_k):
                            index = max_indexes[i]
                            temp_sim = max_sims[i]
                            # u_i = U_I[index][a_item_id]
                            u_i=tf.cond (i_id_In (),
                                              lambda: U_I[index][i_local_index],
                                              tf.Variable (0.0, dtype='float32'))
                            topK_prod.append (temp_sim * u_i)

                        topk_sim_features = [K.expand_dims (sum (topK_prod[:topK])) for topK in
                                             self.max_ks]  # 各个topK下计算的sim积  tensor
                        CF_feature = K.concatenate (topk_sim_features)  # 整合的tensor 形状？

                    return a_user_id, a_item_id, CF_feature  # 同时返回user_id是为了保证输入和输出形状相同，user_id无实质意义

                _1, _2, CF_feature = K.map_fn (fn, (lam_user_id, lam_item_id, lam_mashup_feature))

                return CF_feature

        CF_feature = Lambda (lam) ([user_id, item_id, mashup_feature])

        """看是否需要加入MLP后再整合
        for unit_num in self.text_fc_unit_nums:
            CF_feature = Dense(unit_num, activation='relu')(CF_feature)

        for unit_num in self.tag_fc_unit_nums:
            x = Dense(unit_num, activation='relu')(x)
        """
        predict_vector = concatenate ([CF_feature, x])  # 整合文本和类别特征，尽管层次不太一样
        print (predict_vector.shape)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu') (predict_vector)

        # predict_vector = Flatten()(predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        self.model = Model (
            inputs=[user_id, item_id, user_text_input, item_text_input, user_categories_input, item_categories_input],
            outputs=[predict_result])

        print ('built whole model, done!')
        return self.model

    def get_name(self):
        name = super (gx_text_tag_CF_model, self).get_name ()
        name += self.pmf_01 + '_'
        name += 'max_ks:{} '.format (self.max_ks).replace (',', ' ')
        name += 'predict_fc_unit_nums:{} '.format (self.predict_fc_unit_nums).replace (',', ' ')
        return 'gx_text_tag_CF_model:' + name  # ***

    def get_instances(self, mashup_id_instances, api_id_instances):

        mashup_id2info = self.pd.get_mashup_api_id2info ('mashup')
        api_id2info = self.pd.get_mashup_api_id2info ('api')

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories ('mashup', mashup_id2info, mashup_id, self.Category_type) for
                             mashup_id
                             in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories ('api', api_id2info, api_id, self.Category_type) for api_id in
                          api_id_instances]

        # 针对使用预训练的text_tag_model(embedding复用）
        if self.encoded_texts is None:
            self.process_text ()
        # print('train examples:' + str(len(mashup_id_instances)))
        examples = (
            np.array (mashup_id_instances),
            np.array (api_id_instances),
            np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
            np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
            np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
            np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding'))
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


class gx_text_tag_only_MLP_model (gx_text_tag_model):
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                 inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums,
                 Category_type, tag_manner, merge_manner,
                 mf_embedding_dim, mf_fc_unit_nums,
                 u_factors_matrix, i_factors_matrix, m_id2index,i_id2index, ini_features_array, max_ks, num_feat, topK,CF_self_1st_merge,cf_unit_nums,text_weight=0.5,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[],
                 if_co=1, if_pop=True,co_unit_nums=[1024,256,64,16]):

        super (gx_text_tag_only_MLP_model, self).__init__ (base_dir, remove_punctuation, embedding_name, embedding_dim,
                                                           text_extracter_mode, inception_channels,
                                                           inception_pooling, inception_fc_unit_nums,
                                                           content_fc_unit_nums,
                                                           Category_type, tag_manner, merge_manner, text_fc_unit_nums,
                                                           tag_fc_unit_nums,
                                                           mf_embedding_dim, mf_fc_unit_nums)

        self.u_factors_matrix = u_factors_matrix
        self.i_factors_matrix = i_factors_matrix
        self.ini_features_array = ini_features_array
        self.m_index2id = {index: id for id, index in m_id2index.items ()}
        self.i_id2index = i_id2index



        self.max_ks = max_ks
        self.max_k = max_ks[-1]  # 一定要小于num——users！
        self.predict_fc_unit_nums = predict_fc_unit_nums

        self.num_feat = num_feat
        self._map = None  # pair->x
        self.x_feature_dim = None
        self.mashup_id2CFfeature = None  # mashup-> text,tag 100D
        self.topK = topK
        self.text_weight=text_weight

        self.CF_self_1st_merge=CF_self_1st_merge
        self.cf_unit_nums=cf_unit_nums
        self.model = None

        self.if_co = if_co # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.if_pop = if_pop
        self.co_unit_nums=co_unit_nums
        self.api_id2covec,self.api_id2pop = self.pd.get_api_co_vecs()
        self.mashup_id_pair = self.pd.get_mashup_api_pair('dict')

        self.mashup_id2neighbors={}
        self.mashup_id2CFfeature = {}


    def get_name(self):
        name = super (gx_text_tag_only_MLP_model, self).get_name ()
        cf_='_cf_unit' if self.CF_self_1st_merge  else ''
        name=name+cf_

        co_= '_coInvoke_' + str(self.if_co)
        pop_='_pop_' if self.if_pop else ''

        return 'gx_text_tag_MLP_only_model:' + name+ '_KNN_'+str(self.topK)+'_textWeight_'+str(self.text_weight)+co_+pop_ # ***

    def initialize(self, text_tag_recommend_model, text_tag_model, train_mashup_id_list, train_api_id_list,
                   test_mashup_id_list, test_api_id_list, feature_train_mashup_ids):
        prod = len (test_mashup_id_list) * len (test_mashup_id_list[0])
        D1_test_mashup_id_list = tuple (np.array (test_mashup_id_list).reshape (prod, ))  # 将二维的test数据降维
        D1_test_api_id_list = tuple (np.array (test_api_id_list).reshape (prod, ))

        feature_test_mashup_ids = sorted (list (set (D1_test_mashup_id_list)))  # 测试用mashup的升序排列
        feature_test_api_ids = [0] * len (feature_test_mashup_ids)
        feature_instances_tuple = text_tag_recommend_model.get_instances (feature_test_mashup_ids, feature_test_api_ids,
                                                                          True)  # 只包含mashup信息

        # test样本：提取text和tag feature
        text_tag_middle_model_1 = Model (inputs=[text_tag_model.inputs[0], text_tag_model.inputs[2]],
                                         outputs=[text_tag_model.get_layer ('concatenate_1').input[0],
                                                  text_tag_model.get_layer ('concatenate_1').input[2]])
        text_tag_test_mashup_features = np.hstack (
            text_tag_middle_model_1.predict ([*feature_instances_tuple], verbose=0))  # text，tag 按照mashup id的大小顺序

        # 训练+测试样本  求所有样本的  mashupid，apiid：x
        all_mashup_id_list = train_mashup_id_list + D1_test_mashup_id_list
        all_api_id_list = train_api_id_list + D1_test_api_id_list
        all_instances_tuple = text_tag_recommend_model.get_instances (all_mashup_id_list, all_api_id_list)
        text_tag_middle_model = Model (inputs=text_tag_model.inputs,
                                       outputs=[text_tag_model.get_layer (
                                           'text_tag_feature_extracter').output])  # 输出mashup api的text,tag整合后的特征

        x_features = text_tag_middle_model.predict ([*all_instances_tuple])
        self.x_feature_dim = len (x_features[0])
        self._map = {}  # 基于id
        for index in range (len (x_features)):
            self._map[(all_mashup_id_list[index], all_api_id_list[index])] = x_features[index]

        # 先train，后test mashup id
        all_feature_mashup_ids = feature_train_mashup_ids + feature_test_mashup_ids
        all_features = np.vstack ((self.ini_features_array, text_tag_test_mashup_features))

        # CNN提取的文本特征和tag的embedding大小不一样，所以无法直接拼接计算sim;需要单独计算sim，然后加权求和!!!
        text_dim=self.inception_fc_unit_nums[-1]


        for i in range (len (all_feature_mashup_ids)):  # 为所有mashup找最近
            id2sim = {}
            for j in range (len (feature_train_mashup_ids)):  # 从所有train中找,存放的是内部索引
                if i != j:
                    text_sim = cos_sim (all_features[i][:text_dim], all_features[j][:text_dim])
                    tag_sim = cos_sim (all_features[i][text_dim:], all_features[j][text_dim:])
                    id2sim[j]=  self.text_weight*text_sim+(1- self.text_weight)*tag_sim
            topK_indexes, topK_sims = zip (*(sorted (id2sim.items (), key=lambda x: x[1], reverse=True)[:self.topK]))
            self.mashup_id2neighbors[all_feature_mashup_ids[i]]=[self.m_index2id[index] for index in topK_indexes] #每个mashup距离最近的mashup的id list
            topK_sims = np.array (topK_sims) / sum (topK_sims)
            cf_feature = np.zeros ((self.num_feat))
            for z in range (len (topK_indexes)):
                cf_feature += topK_sims[z] * self.u_factors_matrix[topK_indexes[z]]
            self.mashup_id2CFfeature[all_feature_mashup_ids[i]] = cf_feature

    def get_model(self):
        # 搭建简单模型
        mashup_cf = Input (shape=(self.num_feat,), dtype='float32')
        api_cf = Input (shape=(self.num_feat,), dtype='float32')
        pair_x = Input (shape=(self.x_feature_dim,), dtype='float32')

        co_dim= self.topK if self.if_co == 3 else len(self.api_id2covec) # 3：最近邻是否调用 50D
        co_invoke=Input (shape=(co_dim,), dtype='float32')
        pop=Input (shape=(1,), dtype='float32')

        predict_vector=None
        if self.CF_self_1st_merge:
            predict_vector=concatenate ([mashup_cf, api_cf])
            for unit_num in self.cf_unit_nums:
                predict_vector = Dense (unit_num, activation='relu') (predict_vector)
            predict_vector = concatenate ([predict_vector, pair_x])
        else:
            predict_vector = concatenate ([mashup_cf, api_cf, pair_x])  # 整合文本和类别特征，尽管层次不太一样

        if self.if_co:
            predict_vector1 = Dense (self.co_unit_nums[0], activation='relu') (co_invoke)
            for unit_num in self.co_unit_nums[1:]:
                predict_vector1=Dense (unit_num, activation='relu') (predict_vector1)
            predict_vector = concatenate ([predict_vector,predict_vector1])

        if self.if_pop:
            predict_vector = concatenate ([predict_vector, pop])

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu') (predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        if not (self.if_co or self.if_pop):
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x],outputs=[predict_result])
        elif self.if_co and not self.if_pop:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,co_invoke], outputs=[predict_result])
        elif self.if_pop and not self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,pop], outputs=[predict_result])
        elif self.if_pop and  self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,co_invoke,pop], outputs=[predict_result])

        return self.model

    def get_instances(self, mashup_id_list, api_id_list):
        mashup_cf_features, api_cf_features, x_features, api_co_vecs,api_pops = [], [], [],[],[]
        api_zeros = np.zeros ((self.num_feat))
        num_api=len(self.api_id2covec[0])
        for i in range (len (mashup_id_list)):
            mashup_id = mashup_id_list[i]
            api_id = api_id_list[i]
            mashup_cf_features.append (self.mashup_id2CFfeature[mashup_id])
            api_i_feature = self.i_factors_matrix[self.i_id2index[api_id]] if api_id in self.i_id2index.keys() else api_zeros
            api_cf_features.append (api_i_feature)
            x_features.append (self._map[(mashup_id, api_id)])

            if self.if_co:
                if self.if_co==1:
                    api_co_vecs.append(self.api_id2covec[api_id])
                elif self.if_co==2:
                    api_co_vec=np.zeros((num_api))
                    for m_neigh_id in self.mashup_id2neighbors:
                        for _api_id in self.mashup_id_pair[m_neigh_id]: # 邻居mashup调用过的api
                            api_co_vec[_api_id]=self.api_id2covec[api_id][_api_id]
                    api_co_vecs.append (api_co_vec)
                elif self.if_co == 3: # 是否被最近邻调用
                    api_co_vec = np.zeros ((self.topK))
                    api_co_vec=[1 if api_id in self.mashup_id_pair[m_neigh_id] else 0 for m_neigh_id in self.mashup_id2neighbors[mashup_id]]
                    api_co_vecs.append (api_co_vec)
            if self.if_pop:
                api_pops.append(self.api_id2pop[api_id])

        if not (self.if_co or self.if_pop):
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features)
        elif self.if_co and not self.if_pop:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features), np.array (api_co_vecs)
        elif self.if_pop and not self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features), np.array (api_pops)
        elif self.if_pop and  self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features),np.array(api_co_vecs),np.array(api_pops)


# 搭建模型阶段 抽象tensor的运算
def tensor_sim(f1, f2):
    fenmu = K.sum (tf.multiply (f1, f2))
    sum1 = K.sqrt (K.sum (K.square (f1)))
    sum2 = K.sqrt (K.sum (K.square (f2)))
    return fenmu / (sum1 * sum2)
