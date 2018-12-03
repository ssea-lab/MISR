# -*- coding:utf-8 -*-

import numpy as np
from keras.layers.core import Dropout, Flatten
from keras.layers import Lambda, Concatenate, MaxPooling2D, \
    LSTM, Merge
from keras.layers import Dense, Input, Conv2D, Embedding, concatenate, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import Constant
from keras import initializers
from keras import backend as K

from embedding.embedding import get_embedding_matrix
from helpers.cpt_Sim import get_sims_dict
from process_text.processing_data import process_data
from recommend_models.simple_inception import inception_layer
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size=3


class recommend_Model(object):
    """
    共同基类
    """
    def __init__(self,base_dir,mf_embedding_dim,mf_fc_unit_nums):
        """
        :param base_dir:
        :param mf_embedding_dim: 因为DHSR NCF都包含MF部分，所以这里作为属性（尽管只处理文本部分的模型不需要这部分,此时默认为空即可
        :param mf_fc_unit_nums:
        """
        self.base_dir = base_dir
        self.pd = process_data(self.base_dir, False)
        self.num_users = len(self.pd.get_mashup_api_index2name('mashup'))
        self.num_items = len(self.pd.get_mashup_api_index2name('api'))
        self.mf_embedding_dim=mf_embedding_dim
        self.mf_fc_unit_nums=mf_fc_unit_nums

    def get_name(self):
        pass

    # 类别如何处理？增加一部分？
    def get_model(self):
        """
        **TO OVERIDE**
        :return:  a model
        """
        pass

    def get_merge_MLP(self,input1,input2,MLP_layers):
        """
        难点在于建立model的话，需要设定Input，其中要用到具体形状
        """
        pass

    def get_mf_MLP(self,input_dim1,input_dim2,output_dim,MLP_layers):
        """
        返回id-embedding-merge-mlp的model
        """
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32')
        item_input = Input(shape=(1,), dtype='int32')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=input_dim1, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=input_dim2, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])

        for idx in range(len(MLP_layers)):   # 学习非线性关系
            layer = Dense(MLP_layers[idx],  activation='relu')
            mf_vector = layer(mf_vector)
        model = Model(inputs=[user_input,item_input],outputs=mf_vector)
        return model

    def get_instances(self):
        """
        **TO OVERIDE**
        """
        pass

    def save_sth(self):
        pass


class gx_model(recommend_Model):
    """
    改为抽象基类  只含处理文本
    """

    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim,text_extracter_mode,inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums, mf_embedding_dim=50, mf_fc_unit_nums=[]):
        """
        embedding时需要使用dataset下的所有语料，所以一个base_dir决定一个实例；而数据集划分则是不同的训练实例，跟train有关，跟模型本身无关
        :param base_dir:
        :param embedding_name: 'glove' or 'google_news'
        :param embedding_dim: word embedding 维度
        :param inception_channels: inception 几种 filter的个数；通道数
        :param inception_pooling: inception 最顶层 pooling的形式
        :param inception_fc_unit_nums:  inception之后FC的设置，最后一个决定整个textCNN的特征维度
        :param content_fc_unit_nums: 特征提取部分 FC设置，决定最终维度（整合各部分文本信息
        :param mf_embedding_dim: mf部分每个mashup/id的维度
        :param mf_fc_unit_nums:  mf部分FC设置，决定维度
        :param predict_fc_unit_nums: 最后FC设置
        """
        super(gx_model, self).__init__(base_dir,mf_embedding_dim,mf_fc_unit_nums)

        self.remove_punctuation=remove_punctuation
        self.encoded_texts=None # encoding_padding对象  首次得到embedding层时会初始化

        self.text_extracter_mode=text_extracter_mode

        self.embedding_matrix = None
        self.embedding_layer = None
        self.embedding_name=embedding_name
        self.embedding_dim=embedding_dim

        self.inception_channels=inception_channels
        self.inception_pooling=inception_pooling
        self.inception_fc_unit_nums = inception_fc_unit_nums

        self.content_fc_unit_nums = content_fc_unit_nums  # 在没有使用tag时，可用于整合两个text之后预测***

        self.text_feature_extracter=None # 文本特征提取器

    def get_name(self):
        name=''
        name += 'remove_punctuation:{} '.format(self.remove_punctuation)
        name +='{}_{} '.format(self.embedding_name,self.embedding_dim)
        name += '_{}_ '.format(self.text_extracter_mode)
        name += 'inception_channels:{} '.format(self.inception_channels).replace(',', ' ')
        name += 'inception_pooling:{} '.format(self.inception_pooling)
        name += 'inception_fc_unit_nums:{} '.format(self.inception_fc_unit_nums).replace(',', ' ')
        name += 'content_fc_unit_nums:{} '.format(self.content_fc_unit_nums).replace(',', ' ')
        name += 'mf_embedding_dim:{} '.format(self.mf_embedding_dim)
        name += 'mf_fc_unit_nums:{} '.format(self.mf_fc_unit_nums).replace(',', ' ')
        return 'GX:'+name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    def process_text(self):
        """
        处理文本，先进行;但纯文本和text——tag的处理方法不同，多态
        :return:
        """
        pass

    def get_embedding_layer(self, nonstatic=True):
        """"
        得到定制的embedding层

        paras:
        data_dirs: 存放mashup api 信息的文件夹
        embedding_name：使用哪种pre-trained embedding，google_news or glove
        embedding_path:embedding 文件存放路径
        EMBEDDING_DIM：维度
        nonstatic：基于pre-trained embedding是否微调？
        """

        if self.embedding_layer is None:
            self.process_text()
            # 得到词典中每个词对应的embedding
            num_words = min(MAX_NUM_WORDS, len(self.encoded_texts.word2index))+ 1  # 实际词典大小 +1  因为0代表0的填充向量
            self.embedding_matrix = get_embedding_matrix(self.encoded_texts.word2index, self.embedding_name,
                                                         dimension=self.embedding_dim)
            print('built embedding matrix, done!')

            self.embedding_layer = Embedding(num_words,
                                             self.embedding_dim,
                                             embeddings_initializer=Constant(
                                                 self.embedding_matrix),
                                             input_length=MAX_SEQUENCE_LENGTH,
                                             trainable=nonstatic)  # 定义一层
            print('built embedding layer, done!')
        return self.embedding_layer

    def textCNN_feature_extracter_from_texts(self,embedded_sequences):
        """
        对embedding后的矩阵做textCNN处理提取特征
        :param embedded_sequences:
        :return:
        """

        filtersize_list = [3, 4, 5]
        number_of_filters_per_filtersize = [20,20,20] # 跟50D接近
        pool_length_list = [2, 2, 2]

        conv_list = []
        for index, filtersize in enumerate(filtersize_list):
            nb_filter = number_of_filters_per_filtersize[index]
            pool_length = pool_length_list[index]
            conv = Conv2D(nb_filter=nb_filter, kernel_size=(filtersize,self.embedding_dim), activation='relu')(embedded_sequences)
            pool = MaxPooling2D(pool_size=(pool_length,1))(conv)
            flatten = Flatten()(pool)
            conv_list.append(flatten)

        if (len(filtersize_list) > 1):
            out = Merge(mode='concat')(conv_list)
        else:
            out = conv_list[0]

        return out

    def LSTM_feature_extracter_from_texts(self,embedded_sequences):
        # out=Bidirectional(LSTM(128, implementation=2))(embedded_sequences)
        out = LSTM(128)(embedded_sequences)
        return out

    def SDAE_feature_extracter_from_texts(self):
        pass

    def feature_extracter_from_texts(self):
        """
        对mashup，service的description均需要提取特征，右路的文本的整个特征提取过程
        公用的话应该封装成新的model！
        :param x:
        :return: 输出的是一个封装好的model，所以可以被mashup和api公用
        """
        if self.text_feature_extracter is None: #没求过时
            text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_embedding_layer()  # 参数还需设为外部输入！
            embedded_sequences = embedding_layer(text_input)  # 转化为2D

            if self.text_extracter_mode=='inception' or self.text_extracter_mode=='textCNN':
                # 2D转3D,第三维是channel
                # print(embedded_sequences.shape)
                embedded_sequences = Lambda(lambda x: tf.expand_dims(x, axis=3))(
                    embedded_sequences)  # tf 和 keras的tensor 不同！！！
                print(embedded_sequences.shape)

            if self.text_extracter_mode=='inception':
                x = inception_layer(embedded_sequences,self.embedding_dim,self.inception_channels,self.inception_pooling)  # inception处理
                print('built inception layer, done!')

            elif self.text_extracter_mode=='textCNN':
                x = self.textCNN_feature_extracter_from_texts(embedded_sequences)
            elif self.text_extracter_mode=='LSTM':
                x = self.LSTM_feature_extracter_from_texts(embedded_sequences)
            else:
                raise ValueError('wrong extracter!')

            for FC_unit_num in self.inception_fc_unit_nums[:-1]:
                x = Dropout(0.5)(x)  # concatenate跟FC之间有dropout
                x = Dense(FC_unit_num, activation='relu')(x)

            x = Dropout(0.5)(x)  # concatenate跟FC之间有dropout
            x = Dense(self.inception_fc_unit_nums[-1], activation='relu',name='text_feature_extracter')(x) #结构之后的最后一个FC层命名

            self.text_feature_extracter=Model(text_input, x)
        return self.text_feature_extracter

    def get_categories_feature_extracter(self):
        """
        tag特征提取
        :return:
        """
        pass

    def get_text_tag_part(self):
        """
        整合文本和tag
        :return:
        """
        pass

    def get_model(self):
        pass

    def get_instances(self,mashup_id_instances, api_id_instances):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！
        :param args:
        :return:
        """
        pass


class DHSR_model(recommend_Model):
    def __init__(self,base_dir, mf_embedding_dim,feature_size=8,layers1=[32,16,8],layers2=[32,16,8]):
        super(DHSR_model, self).__init__(base_dir, mf_embedding_dim,layers1)
        self.sims_dict = get_sims_dict(False,True) # 相似度对象，可改参数？
        self.feature_size=feature_size
        self.layers2 = layers2

    def get_name(self):
        return '_DHSR'

    def get_model(self):
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        text_input = Input(shape=(self.feature_size,), dtype='float32', name='text_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])  # element-wise multiply    ???

        for idx in range(len(self.mf_fc_unit_nums)):   # 学习非线性关系
            layer = Dense(self.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx)
            mf_vector = layer(mf_vector)

        # Text part
        # text_input = Dense(10, activation='relu', kernel_regularizer=l2(0.01))(text_input)  #   sim? 需要再使用MLP处理下？

        # Concatenate MF and TEXT parts
        predict_vector = concatenate([mf_vector, text_input])

        for idx in range(len(self.layers2)):   # 整合后再加上MLP？
            layer = Dense(self.layers2[idx],  activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = Dropout(0.5)(predict_vector)    # 使用dropout?

        # Final prediction layer
        predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=[user_input, item_input, text_input],outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        examples = (np.array(mashup_id_instances),np.array(api_id_instances),np.array(sims))
        return examples

    def save_sth(self):
        self.sims_dict.save_sims_dict()


class DHSR_noMF(DHSR_model):
    def get_name(self):
        return '_DHSR_noMF'

    def get_model(self):
        # Input Layer
        text_input = Input(shape=(self.feature_size,), dtype='float32', name='text_input')

        predict_vector= Dense(self.layers2[0],  activation='relu')(text_input)

        for idx in range(len(self.layers2))[1:]:   # 整合后再加上MLP？
            layer = Dense(self.layers2[idx],  activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = Dropout(0.5)(predict_vector)    # 使用dropout?

        # Final prediction layer
        predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=text_input,outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        returns=[]
        returns.append(sims)
        return np.array(returns)


class NCF_model(recommend_Model):
    def __init__(self, base_dir, mf_embedding_dim, mf_layers=[64, 32, 16, 8], reg_layers=[0.01, 0.01, 0.01, 0.01], reg_mf=0.01):
        super(NCF_model, self).__init__(base_dir, mf_embedding_dim, mf_layers)
        assert len(mf_layers) == len(reg_layers)
        self.reg_layers=reg_layers
        self.reg_mf=reg_mf
        self.name = '_NCF'

    def get_model(self):
        num_layer = len(self.layers)  # Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.reg_mf), input_length=1) #

        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.reg_mf), input_length=1) #

        MLP_Embedding_User = Embedding(input_dim=self.num_users, output_dim=int(self.mf_fc_unit_nums[0] / 2), name="mlp_embedding_user",
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.reg_layers[0]), input_length=1) #

        MLP_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=int(self.mf_fc_unit_nums[0] / 2), name='mlp_embedding_item',
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.reg_layers[0]), input_length=1) #

        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        #   mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
        mf_vector=Multiply()([mf_user_latent, mf_item_latent])

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        #   mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

        for idx in range(1, num_layer):
            layer = Dense(self.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx) # kernel_regularizer=l2(reg_layers[idx]),
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
        # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
        #   predict_vector = merge([mf_vector, mlp_vector], mode='concat')
        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(input=[user_input, item_input],output=prediction)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        examples = (np.array(mashup_id_instances),np.array(api_id_instances))
        return examples


