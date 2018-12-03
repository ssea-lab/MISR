# -*- coding:utf-8 -*-

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
keras_filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'

class encoding_padding(object):
    """
    预处理文本，根据词频对词汇编码，得到文本（新，旧）的词汇index形式的编码
    """
    def __init__(self,descriptions,remove_punctuation):
        self.word2index =None
        self.texts_in_index=None
        self.process_text(descriptions,remove_punctuation)

    def process_text(self,descriptions,remove_punctuation):
        """
        process descriptions
        默认按照文本中词频建立词典   0空白 1 最高频词汇 ....！
        :return:
        """

        print('Found %s texts.' % len(descriptions))
        filters= keras_filters if remove_punctuation else '' # 是否过滤标点
        tokenizer = Tokenizer(filters=filters, num_words=MAX_NUM_WORDS)  # 声明最大长度，默认使用词频***
        tokenizer.fit_on_texts(descriptions)
        # vectorize the text samples into a 2D integer tensor
        sequences = tokenizer.texts_to_sequences(descriptions)

        # 字典，将单词（字符串）映射为索引
        self.word2index = tokenizer.word_index
        print('word2index:')
        # print(self.word2index)
        """
        with open('../data/word2index','w',encoding='utf-8') as f:
            for word,index in self.word2index.items():
                f.write('{} {}\n'.format(word,index))
        """
        print('Found %s unique tokens.' % len(self.word2index))

        self.texts_in_index = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 开头用0（index）填充

        """
        np.savetxt('../data/keras_encoding_texts',self.texts_in_index,fmt='%d')
        print('save keras_encoding_texts,done!')
        """

    def get_texts_in_index(self, alist,manner,start_index=0):
        """
        得到多个文本的词汇index编码形式
        :param alist:
        :param manner:
        :param start_index:
        :return:
        """
        if manner=='keras_setting':
            return [self.get_text_in_index1(start_index,id) for id in alist] # 一维列表，每个值是mashup，api id
        elif manner=='self_padding':
            return [self.get_text_in_index2(word_list) for word_list in alist] # 此时输入二维list 每一行是一个文本的词列表
        else:
            raise ValueError('wrong manner!')
        return 0

    def get_text_in_index1(self,start_index,id):
        """
        根据keras的文本预处理技术得到的各个des的padding
        :param mashup_api:
        :param id:
        :return:
        """
        return self.texts_in_index[start_index+id]

    def get_text_in_index2(self,word_list):
        """
        根据 word2index将文本转化为index并padding
        :param word_list:
        :return:
        """
        result=[0]*MAX_SEQUENCE_LENGTH
        size=len(word_list)
        _sub=MAX_SEQUENCE_LENGTH-size
        indexs=[i for i in range(size)][::-1] # 反向开始变换
        for i in indexs:
            word_index=self.word2index.get(word_list[i])
            if word_index is not None:
                result[_sub+i]=word_index
        return result


