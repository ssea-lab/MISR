# -*- coding:utf-8 -*-
import os

import gensim
import numpy as np

glove_embedding_path = r'../data/pre_trained_embeddings'
google_embedding_path = r''


def get_word_embedding(embedding, embedding_name, word,dimension,initize='zeros'):
    """
    得到一个词的embedding向量，不存在为0，或者按-切割再相加
    :param embedding:
    :param embedding_name:
    :param word:
    :param dimension:
    :param initize 可以选择不存在词的初始方式 随机化还是0
    :return:
    """
    if embedding_name == 'glove':
        vector = embedding.get(word)
    elif embedding_name == 'google_news':
        vector = embedding.get_vector(word)

    if vector is not None: # 存在该词
        sum_vec = vector
    else:
        sum_vec = np.zeros(dimension)
        subs = word.split('-')  # 用-连接的，vector累加***
        if len(subs) > 1:
            for sub_word in subs:
                sub_vector = get_word_embedding(embedding, embedding_name, sub_word,dimension)
                if sub_vector is None:  # 子字符串不存在
                    continue
                sum_vec = sum_vec + sub_vector
        if initize == 'random' and (sum_vec==np.zeros(dimension)).all(): # 如果不存在且切割后的词仍不存在
            sum_vec=np.random.random(dimension)
    return sum_vec


def get_embedding(embedding_name, dimension=50):
    """
    读取预训练的embedding模型
    :param embedding_name:
    :param dimension:
    :return:
    """
    if embedding_name == 'google_news':
        embedding = gensim.models.KeyedVectors.load_word2vec_format(google_embedding_path, binary=True)
    elif embedding_name == 'glove':
        embedding = {}
        with open(os.path.join(glove_embedding_path,
                               "glove.6B.{}d.txt".format(dimension)), encoding='utf-8') as f:  # dict: word->embedding(array)
            for line in f:
                values = line.split()
                word = values[0]
                embedding[word] = np.asarray(values[1:], dtype='float32')
    print('Found %s word vectors in pre_trained embedding.' % len(embedding))
    return embedding


def get_embedding_matrix(word2index, embedding_name, dimension=50):
    """
    得到特定语料库词典对应的embedding矩阵
    :param word2index: 本任务语料中的词
    :param embedding_name:
    :param embedding_path:
    :param dimension:
    :return: 2D array
    """
    embedding=get_embedding(embedding_name, dimension)
    # construct an embedding_matrix
    num_words = len(word2index)+1  # 实际词典大小 +1
    embedding_matrix = np.zeros((num_words, dimension)) 
    for word, index in word2index.items(): # keras 文本预处理后得到的字典 按词频 对单词编码
        embedding_matrix[index]=get_word_embedding(embedding, embedding_name, word, dimension)
    return embedding_matrix
