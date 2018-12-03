from recommend_models.recommend_Model import gx_model
import numpy as np
from keras.layers.core import Dropout
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras import backend as K

from embedding.encoding_padding_texts import encoding_padding

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size=3


class gx_text_only_model(gx_model):
    """"只处理text 不处理 tag的结构;但不加入MF部分"""

    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode, inception_channels,
                 inception_pooling, inception_fc_unit_nums, content_fc_unit_nums, mf_embedding_dim=50, mf_fc_unit_nums=[]):

        super(gx_text_only_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,
                                                 inception_channels, inception_pooling, inception_fc_unit_nums,
                                                 content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums)

    def get_name(self):
        name =super(gx_text_only_model, self).get_name()
        return 'text_only_model:' + name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    def process_text(self):
        """
        只处理文档  不处理tag
        :param data_dirs: 某个路径下的mashup和api文档
        :return:
        """
        mashup_descriptions, api_descriptions, mashup_categories, api_categories=self.pd.get_all_texts()
        descriptions = mashup_descriptions+api_descriptions # 先mashup后api 无tag

        self.encoded_texts=encoding_padding(descriptions,self.remove_punctuation) # 可得到各文本的encoded形式

    def get_text_tag_part(self, user_text_input, item_text_input):
        """
        只处理文本的结构
        """
        user_text_feature = self.feature_extracter_from_texts()(user_text_input) #(None,1,embedding)?
        item_text_feature = self.feature_extracter_from_texts()(item_text_input)
        x = concatenate([user_text_feature, item_text_feature])
        for unit_num in self.content_fc_unit_nums: # 整合text
            x = Dense(unit_num, activation='relu')(x)
        return x

    def get_model(self):
        user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

        predict_vector = self.get_text_tag_part(user_text_input, item_text_input)
        # 只处理文本时 merge feature后已经全连接，所以不要predict——fc
        predict_vector = Dropout(0.5)(predict_vector)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)
        model = Model(inputs=[user_text_input, item_text_input],outputs=[predict_result])

        print('built whole model, done!')
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        """
        """
        examples=(
        np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
        np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
        )

        return examples


class gx_text_only_MF_model(gx_text_only_model):
    """
    只处理text+MF
    """
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,inception_channels, inception_pooling,
                 inception_fc_unit_nums, content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums, predict_fc_unit_nums):
        super(gx_text_only_MF_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim,text_extracter_mode,
                                                    inception_channels, inception_pooling, inception_fc_unit_nums,
                                                    content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums)
        self.predict_fc_unit_nums=predict_fc_unit_nums # 用于整合文本和mf之后的预测

    def get_name(self):
        name = super(gx_text_only_model, self).get_name()
        name += 'predict_fc_unit_nums:{} '.format(self.predict_fc_unit_nums).replace(',', ' ');
        return 'gx_text_only_MF_model:' + name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    def get_model(self):
        user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')
        x = self.get_text_tag_part(user_text_input, item_text_input)
        # x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        # print(x.shape) # (?, 1, 50)

        # left part
        user_id = Input(shape=(1,), dtype='int32', name='user_input') # 一个数字
        item_id = Input(shape=(1,), dtype='int32', name='item_input')
        # y = self.get_mf_part(user_id, item_id)

        # MLP part  使用接口model
        mf_mlp = self.get_mf_MLP(self.num_users, self.num_items, self.mf_embedding_dim, self.mf_fc_unit_nums)
        y= mf_mlp([user_id,item_id])

        #print(y.shape) # (?, 1, 50)

        # merge the two parts
        predict_vector = concatenate([x, y])
        print('final merge,done!')
        print(predict_vector.shape) # (?, 1, 100)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense(unit_num, activation='relu')(predict_vector)

        # predict_vector=Flatten()(predict_vector)
        predict_vector = Dropout(0.5)(predict_vector)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)
        model = Model(inputs=[user_id,item_id,user_text_input, item_text_input], outputs=[predict_result])

        print('built whole model, done!')
        return model

    def get_instances(self, mashup_id_instances, api_id_instances):
        examples=(
        np.array(mashup_id_instances),
        np.array(api_id_instances),
        np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
        np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
        )

        return examples
