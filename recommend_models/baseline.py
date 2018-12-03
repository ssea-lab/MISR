import heapq
import os
import sys

from main.para_setting import Para
from process_text.processing_data import process_data

sys.path.append("..")
from main.evalute import evalute, summary

sys.path.append("..")
from gensim.corpora import Dictionary
from gensim.models import HdpModel, TfidfModel
import numpy as np
from helpers.util import cos_sim


class gensim_data(object):
    def __init__(self,mashup_descriptions, api_descriptions, mashup_categories=None, api_categories=None,tag_times=1):

        # text+tag转为[['1','1'],[]]
        if mashup_categories is not None:
            self.mashup_dow = np.vstack ((mashup_descriptions, mashup_categories))
            for i in range(tag_times-1):
                self.mashup_dow = np.vstack ((self.mashup_dow, mashup_categories)) # 直接将文本和tag拼接，是否有更好的方法？增加出现次数？
        self.mashup_dow = [[str (index) for index in indexes] for indexes in self.mashup_dow]

        if api_categories is not None:
            self.api_dow = np.vstack ((mashup_descriptions, api_descriptions))
            for i in range(tag_times-1):
                self.api_dow = np.vstack ((self.api_dow, api_descriptions))
        self.api_dow = [[str (index) for index in indexes] for indexes in self.api_dow]

        # 字典和编码后的info
        self.dct = Dictionary (self.mashup_dow + self.api_dow)
        self.mashup_dow = [self.dct.doc2bow (mashup_info) for mashup_info in self.mashup_dow]
        self.api_dow = [self.dct.doc2bow (api_info) for api_info in self.api_dow]

        self.num_topics =0
        self.model = None
        self._mashup_hdp_features= None
        self._api_hdp_features= None

    # 只关注词在文本中是否出现过，二进制，用于计算cos和jaccard
    def get_binary_v(self,all_mashup_num,all_api_num):
        dict_size=len(self.dct)
        mashup_binary_matrix=np.zeros((all_mashup_num,dict_size))
        api_binary_matrix = np.zeros ((all_api_num, dict_size))
        mashup_words_list=[] # 所有出现过的词
        api_words_list = []
        for i in range(all_mashup_num):
            temp_words_list,_=zip(*self.mashup_dow[i])
            mashup_words_list.append(temp_words_list)
            for j in temp_words_list:# 出现的index
                mashup_binary_matrix[i][j]=1.0

        for i in range(all_api_num):
            temp_words_list,_=zip(*self.api_dow[i])
            api_words_list.append(temp_words_list)
            for j in temp_words_list:# 出现的index
                api_binary_matrix[i][j]=1.0
        return mashup_binary_matrix,api_binary_matrix,mashup_words_list,api_words_list

    def model_pcs(self,model_name,all_mashup_num,all_api_num):
        # 按照0-all——num得到的其实是按真实id的映射！！！
        # hdp结果形式：[(0, 0.032271167132309014),(1, 0.02362695056720504)]
        if model_name=='HDP':
            self.model = HdpModel(self.mashup_dow+self.api_dow, self.dct)
            self.num_topics = self.model.get_topics ().shape[0]
        elif model_name=='TF_IDF':
            self.model =TfidfModel (self.mashup_dow+self.api_dow)
            self.num_topics=len(self.dct)
        else:
            raise ValueError('wrong gensim_model name!')

        mashup_hdp_features=[self.model[mashup_info] for mashup_info in self.mashup_dow]
        api_hdp_features = [self.model[api_info] for api_info in self.api_dow]

        self._mashup_hdp_features=np.zeros((all_mashup_num,self.num_topics))
        self._api_hdp_features = np.zeros((all_api_num, self.num_topics))
        for i in range(all_mashup_num):
            for index,value in mashup_hdp_features[i]:
                self._mashup_hdp_features[i][index]=value
        for i in range(all_api_num):
            for index,value in api_hdp_features[i]:
                self._api_hdp_features[i][index]=value
        return self._mashup_hdp_features,self._api_hdp_features


# ***处理数据等最好不要放在recommend类中，并且该方法应设为recommend的子类？***
def Samanta(text_tag_recommend_model, topK,if_pop=False):
    """
    :param Para:
    :param text_tag_recommend_model: 基于该model的基本数据
    :param topK:
    :return:
    """

    api2pop=None
    if if_pop:
        api_co_vecs, api2pop = text_tag_recommend_model.pd.get_api_co_vecs ()

    test_mashup_num = len(Para.test_mashup_id_list)

    all_mashup_num=len(text_tag_recommend_model.pd.get_mashup_api_index2name('mashup'))
    all_api_num = len(text_tag_recommend_model.pd.get_mashup_api_index2name('api'))

    mashup_hdp_path=os.path.join(Para.data_dir, 'mashup_hdp.txt')
    api_hdp_path = os.path.join(Para.data_dir, 'api_hdp.txt')

    # 获取mashup_hdp_features,api_hdp_features
    if not os.path.exists(api_hdp_path):
        # text,tag在encoding之后的向量，array形式
        gd=gensim_data(*text_tag_recommend_model.get_instances([i for i in range(all_mashup_num)], [i for i in range(all_api_num)],False))
        _mashup_hdp_features,_api_hdp_features=gd.model_pcs('HDP',all_mashup_num,all_api_num)
        np.savetxt(mashup_hdp_path,_mashup_hdp_features)
        np.savetxt(api_hdp_path, _api_hdp_features)
    else:
        _mashup_hdp_features=np.loadtxt(mashup_hdp_path)
        _api_hdp_features=np.loadtxt(api_hdp_path)

    candidate_ids_list = []
    all_predict_results=[]
    for i in range(test_mashup_num):
        test_mashup_id=Para.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = Para.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        id2sim={}
        for local_train_mashup_index in range(len(Para.feature_train_mashup_ids)): # u_factors_matrix要用局部索引
            id2sim[local_train_mashup_index]=cos_sim(_mashup_hdp_features[test_mashup_id],_mashup_hdp_features[Para.feature_train_mashup_ids[local_train_mashup_index]])
        topK_indexes,topK_sims=zip(*(sorted(id2sim.items(), key=lambda x: x[1], reverse=True)[:topK]))
        topK_sims=np.array(topK_sims)/sum(topK_sims)
        cf_feature=np.zeros((Para.num_feat))
        for z in range(len(topK_indexes)):
            cf_feature+= topK_sims[z] * Para.u_factors_matrix[topK_indexes[z]]

        predict_results = []
        temp_predict_results=[] # 需要用pop进行重排序时的辅助
        api_zeros=np.zeros((Para.num_feat))
        for api_id in candidate_ids: # id
            api_i_feature= Para.i_factors_matrix[Para.i_id2index[api_id]] if api_id in Para.i_id2index.keys() else api_zeros  # 可能存在测试集中的api不在train中出现过的场景
            cf_score=np.sum(np.multiply(api_i_feature, cf_feature))
            sim_score=cos_sim(_mashup_hdp_features[test_mashup_id],_api_hdp_features[api_id])
            if if_pop:
                temp_predict_results.append((api_id,cf_score*sim_score))
            else:
                predict_results.append(cf_score*sim_score)

        if if_pop:
            max_k_pairs = heapq.nlargest (topK, temp_predict_results, key=lambda x: x[1])  # 根据score选取topK
            max_k_candidates, _ = zip (*max_k_pairs)
            max_k_candidates=set(max_k_candidates)
            predict_results=[api2pop[api_id] if api_id in max_k_candidates else -1 for api_id in candidate_ids] # 重排序

        all_predict_results.append(predict_results)
    print('Samanta test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, Para.grounds, Para.topKs)  # 评价
    _name='_pop' if if_pop else ''
    csv_table_name = Para.data_name + 'Samanta_model'+_name + "\n"   # model.name
    summary(Para.evaluate_path, csv_table_name, evaluate_result, Para.topKs)  # 记录


def TF_IDF(text_tag_recommend_model):
    """
    可以跟写到Samanta的类中，但太混乱，没必要
    :return:
    """
    all_mashup_num=len(text_tag_recommend_model.pd.get_mashup_api_index2name('mashup'))
    all_api_num = len(text_tag_recommend_model.pd.get_mashup_api_index2name('api'))

    gd = gensim_data (*text_tag_recommend_model.get_instances ([i for i in range (all_mashup_num)], [i for i in range (all_api_num)],
                                                False))
    _mashup_IFIDF_features, _api_IFIDF_features = gd.model_pcs ('TF_IDF',all_mashup_num, all_api_num)

    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(Para.test_mashup_id_list)):
        test_mashup_id=Para.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = Para.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(_mashup_IFIDF_features[test_mashup_id],_api_IFIDF_features[api_id])
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('TF_IDF test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, Para.grounds, Para.topKs)  # 评价
    csv_table_name = Para.data_name + 'TF_IDF' + "\n"   # model.name
    summary(Para.evaluate_path, csv_table_name, evaluate_result, Para.topKs)  # 记录


def binary_keyword(text_tag_recommend_model):
    # pop

    all_mashup_num=len(text_tag_recommend_model.pd.get_mashup_api_index2name('mashup'))
    all_api_num = len(text_tag_recommend_model.pd.get_mashup_api_index2name('api'))
    api_co_vecs, api2pop = text_tag_recommend_model.pd.get_api_co_vecs ()

    gd = gensim_data (*text_tag_recommend_model.get_instances ([i for i in range (all_mashup_num)], [i for i in range (all_api_num)],
                                                False))
    mashup_binary_matrix, api_binary_matrix, mashup_words_list, api_words_list = gd.get_binary_v (all_mashup_num, all_api_num)

    # 测试WVSM(Weighted Vector Space Model)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(Para.test_mashup_id_list)):
        test_mashup_id=Para.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = Para.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(mashup_binary_matrix[test_mashup_id],api_binary_matrix[api_id])*api2pop[api_id]
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WVSM test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, Para.grounds, Para.topKs)  # 评价
    csv_table_name = Para.data_name + 'WVSM' + "\n"   # model.name
    summary(Para.evaluate_path, csv_table_name, evaluate_result, Para.topKs)  # 记录

    # 测试WJaccard(Weighted Jaccard)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(Para.test_mashup_id_list)):
        test_mashup_id=Para.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = Para.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            mashup_set=set(mashup_words_list[test_mashup_id])
            api_set = set (api_words_list[api_id])
            sim_score=1.0*len(mashup_set.intersection(api_set))/len(mashup_set.union(api_set))*api2pop[api_id]
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WJaccard test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, Para.grounds, Para.topKs)  # 评价
    csv_table_name = Para.data_name + 'WJaccard' + "\n"   # model.name
    summary(Para.evaluate_path, csv_table_name, evaluate_result, Para.topKs)  # 记录
