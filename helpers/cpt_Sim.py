# -*- coding:utf-8 -*-
import os
import pickle
from math import log
import random

import numpy as np
from process_text.processing_data import process_data
from embedding.embedding import get_word_embedding, get_embedding
from helpers.util import cos_sim

stop_words = set(['!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'])  # 去标点符号？？
data_dir = r'../data'


class cpt_Sim(object):
    def __init__(self, data_dir, embedding_name, embedding_dim, tag_coefficient, k, b, weighted_intervals,
                 unweighted_intervals, process_new,remove_stopwords):
        """
        :param self:
        :param tag_coefficient: tag amplification coefficient
        :param k:smooth parameter
        :param b: smooth parameter
        :param weighted_intervals:
        :param unweighted_intervals:
        :return:
        """
        self.process_new = process_new
        self.data_dir = data_dir
        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.tag_coefficient = int(tag_coefficient)
        self.k = k
        self.b = b
        self.weighted_intervals = weighted_intervals
        self.unweighted_intervals = unweighted_intervals

        self.num_mashup = 0
        self.num_api = 0
        self.word2inedx = {}  # 词到index映射  在该类中主要以index形式存在  作key时比str节省内存
        self.wordindex2IDF = {}
        self.average_len = 0
        self.stopwords= stop_words if remove_stopwords else set()

        self.mashup_descriptions = None
        self.api_descriptions = None
        self.wordindex2embedding = {} # 词index对应的embedding

        self.words_Sim = {}  # 词对间的cos sim  不需求全部  随用随求并保存  {(,):float,}
        self.mashup2api_Sim = {}  # mashup 到 api 的sim  id形式 {(,):float,}

        self.initialize_sims_dict()

    def initialize_sims_dict(self):
        """
        返回数据集中所有mashup，api对的sim  [] 维度数随weighted_intervals,nweighted_intervals变化
        :return:
        """
        path = self.data_dir + '/DHSR.sim'

        if not self.process_new and os.path.exists(path):
            with open(path, 'rb') as f:
                self.sims_dict = pickle.load(f)
        else: # 需要求的时候
            self.sims_dict ={}

    def save_sims_dict(self):  # 最后要将模型保存
        path = self.data_dir + '/DHSR.sim'
        with open(path, 'wb') as f:
            pickle.dump(self.sims_dict, f)

    def get_data(self):
        """
        获得文本，统计词汇，对文本用词index重新编码，获得词（index)的embedding
        :return:
        """
        pd = process_data(self.data_dir, False)
        mashup_descriptions, api_descriptions, mashup_categories, api_categories = pd.get_all_texts()
        self.num_mashup = len(mashup_descriptions)
        self.num_api = len(api_descriptions)
        # 整合文本
        for index in range(self.num_mashup):
            for i in range(self.tag_coefficient):
                mashup_descriptions[index] += mashup_categories[index]
        for index in range(self.num_api):
            for i in range(self.tag_coefficient):
                api_descriptions[index] += api_categories[index]

        # 统计字符串 并统计IDF
        word2DF = {}  # 词的出现mashup/api 的set  word->set

        word_count = 0
        for text_index in range(self.num_mashup):  # 记录的是mashup inedx
            mashup_text = mashup_descriptions[text_index].split()
            word_count += len(mashup_text)
            for word in mashup_text:
                if word not in self.stopwords and word not in self.word2inedx.keys():  # ？？？去标点符号？？
                    word2DF[word] = set()  # word2DF和word2inedx的key是同步更新的
                    self.word2inedx[word] = len(self.word2inedx)  # 词到index的索引,新词加在末尾
                word2DF[word].add(text_index)

        for text_index in range(self.num_api):  # 记录的是mashup index
            api_text = api_descriptions[text_index].split()
            word_count += len(api_text)
            true_index = text_index + self.num_mashup
            for word in api_text:
                if word not in self.stopwords and word not in self.word2inedx.keys():
                    word2DF[word] = set()
                    self.word2inedx[word] = len(self.word2inedx)  # 词到index的索引
                word2DF[word].add(true_index)

        # 将mashup_descriptions 转化为 word index的形式
        self.mashup_descriptions = [[self.word2inedx.get(word) for word in text.split()] for text in mashup_descriptions]
        self.api_descriptions = [[self.word2inedx.get(word) for word in text.split()] for text in api_descriptions]
        # print(mashup_descriptions) # self.
        # print(api_descriptions)
        # print(self.word2inedx)

        # 计算IDF
        num_all_texts = self.num_mashup + self.num_api
        self.average_len = word_count / num_all_texts
        self.wordindex2IDF = {self.word2inedx.get(word): log(num_all_texts / len(existed_docs)) for word, existed_docs
                              in word2DF.items()}

        # 获得每个词的embedding: id->array
        embedding = get_embedding(self.embedding_name, self.embedding_dim)
        self.wordindex2embedding = {
            self.word2inedx.get(word): get_word_embedding(embedding, self.embedding_name, word, self.embedding_dim,
                                                          initize='random') for word in word2DF.keys()}
        # print(self.wordindex2embedding)

    def get_mashup_api_sim(self,mashup_id, api_id):
        """
        因为要计算的sim未知，所以实时计算并存储
        外部接口  有则直接返回；无则计算并存储
        :param mashup_id:
        :param api_id:
        :return:
        """
        sim = self.sims_dict.get((mashup_id, api_id))
        if sim is None:
            if self.mashup_descriptions is None: # 求的时候有必要先初始化各种结构
                self.get_data()
            mashup_text_index = self.mashup_descriptions[mashup_id] # index形式
            api_text_index = self.api_descriptions[api_id]
            # print(api_text_index)
            if len(mashup_text_index) == 0 or len(api_text_index) == 0:  # 对于不存在信息的mashup和api怎么处理？ 假设sim为【-1,1】上的n-D随机数
                total_dim=len(self.weighted_intervals)+len(self.unweighted_intervals)-1
                sim=[random.uniform(-1,1) for i in range(total_dim)]
            else:
                sim_weighted = self.get_weighted(mashup_text_index, api_text_index)
                # print('get_weighted,done!')
                # print(sim_weighted)
                sim_unweighted = self.get_unweighted(mashup_text_index, api_text_index)
                # print('get_unweighted,done!')
                # print(sim_unweighted)
                sim_mean = self.get_mean(mashup_text_index, api_text_index)
                # print('get_mean,done!')
                # print(sim_mean)
                sim = sim_weighted + sim_unweighted
                sim.append(sim_mean)
            self.sims_dict[(mashup_id, api_id)] = sim
        # print(sim)
        return sim

    def get_weighted(self, mashup_text_index, api_text_index):
        if len(mashup_text_index) >= len(api_text_index):
            long_text = mashup_text_index
            short_text = api_text_index
        else:
            long_text = api_text_index
            short_text = mashup_text_index

        intervals = []  # 暂存各个fe的list：[[],[],[],[]]
        for i in range(len(self.weighted_intervals) - 1):
            intervals.append([])

        for w in long_text:  # index
            sems = [self.cpt_wod_cos_sim(w, t) for t in short_text]
            # print(sems)
            sem = max(sems)
            interval_index = choose_a_interal(self.weighted_intervals, sem)
            # print('sem:{},index:{}'.format(sem,interval_index))
            fe = self.wordindex2IDF.get(w) * sem * (self.k + 1) / (sem + self.k * (1 - self.b + self.b * len(short_text) / self.average_len))
            intervals[interval_index].append(fe)

        final_fes = []  # 最终的sim list
        for a_interval in intervals:
            if len(a_interval) == 0:
                final_fes.append(0)
            else:
                final_fes.append(sum(a_interval) / len(a_interval))
        return final_fes

    def get_unweighted(self, mashup_text_index, api_text_index):
        word_sims = [[self.cpt_wod_cos_sim(word1, word2) for word2 in api_text_index] for word1 in mashup_text_index]

        intervals = []  # 暂存各个fe的list：[[],[],[],[]]
        for i in range(len(self.unweighted_intervals) - 1):
            intervals.append([])

        for word_sim_list in word_sims:
            # print(word_sim_list)
            for word_sim in word_sim_list:
                interval_index = choose_a_interal(self.unweighted_intervals, word_sim)
                intervals[interval_index].append(word_sim)

        final_fes = []  # 最终的sim list
        for a_interval in intervals:
            if len(a_interval) == 0:
                final_fes.append(0)
            else:
                final_fes.append(sum(a_interval) / len(a_interval))
        return final_fes

    def get_mean(self, mashup_text_index, api_text_index):
        embedding1 = np.array([self.wordindex2embedding.get(word) for word in mashup_text_index]).mean(axis=0)
        embedding2 = np.array([self.wordindex2embedding.get(word) for word in api_text_index]).mean(axis=0)
        sim = cos_sim(embedding1, embedding2)
        return sim

    def cpt_wod_cos_sim(self, id1, id2):
        """
        计算词（id）间的sim，并存储供索引
        :param id1:
        :param id2:
        :return:
        """

        if id1 == id2:
            return 1
        id_b = max(id1, id2)
        id_s = min(id1, id2)
        value = self.words_Sim.get((id_s, id_b))  # 小到大，按顺序
        if value is None:
            value = cos_sim(self.wordindex2embedding.get(id_s), self.wordindex2embedding.get(id_b))
            self.words_Sim[(id_s, id_b)] = value
        return value


def choose_a_interal(intervals, value):
    """
    根据intervals和值判断纳入哪个区间，返回区间index
    :param intervals:
    :param value:
    :return:
    """

    for index in range(len(intervals) - 1):
        if value >= intervals[index] and value < intervals[index + 1]:
            return index
    if value >= intervals[-1]: # 不在以上区间内，为什么，∞？ 浮点数计算原因，有的是1.00000000x
        return len(intervals) - 2
    return len(intervals) - 2


def get_sims_dict(process_new,remove_stopwords):
    embedding_name = 'glove'
    embedding_dim = 50
    tag_coefficient = 2
    k = 1.2
    b = 0.75
    weighted_intervals = [-1, 0.15, 0.4, 0.8, 1]
    unweighted_intervals = [-1, 0.45, 0.8, 1]
    cs = cpt_Sim(data_dir, embedding_name, embedding_dim, tag_coefficient, k, b, weighted_intervals,
                 unweighted_intervals, process_new,remove_stopwords)
    return cs # 返回对象


if __name__ == '__main__':
    get_sims_dict()
