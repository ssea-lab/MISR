import os
import sys
sys.path.append("..")
from mf.get_UI import get_UV
from spilt_dataset.new_split_dataset import split_dataset, transfer_UI_matrix


import numpy as np

class Para(object):

    root_path=os.path.abspath('..')   #表示当前所处的文件夹的绝对路径
    features_result_path= os.path.join(root_path,'feature')
    history_result_path= os.path.join(root_path,'history')
    data_dir = os.path.join(root_path, 'data')
    evaluate_path = os.path.join(data_dir, 'evaluate_result.csv')

    num_epochs =5   # 5 7 10
    small_batch_size = 16
    big_batch_size = 128
    learning_rate = 0.001

    topKs = [k for k in range(5, 55, 5)] # 测试NDCG用

    remove_punctuation = True
    embedding_name = 'glove'
    embedding_dim = 50
    # text_extracter_mode= 'textCNN'# 'inception','LSTM','textCNN'
    inception_channels = [10, 10, 10, 20, 10]
    inception_pooling = 'global_max'  # 'global_max' 'max' 'global_avg'
    inception_fc_unit_nums = [100, 50]  # inception后接的FC
    content_fc_unit_nums = [100, 50]  # 所有文本相关的维度 #  [256,64,16,8]  #4*50 merge [200,100,50]  # MLP 50  50*2 merge [100,50]

    text_fc_unit_nums = [100, 50]  # 文本向量和类别向量分别MLP后merge再MLP # [100,50]
    tag_fc_unit_nums = [100, 50]  # [100,50]

    mf_embedding_dim = 50  # 8 50
    mf_fc_unit_nums = [120, 50]  # mf部分最终维度 #  32,16,8   120,50

    predict_fc_unit_nums = [200, 100, 50, 25]  # 整合后的最后维度 #  32,16,8   200,100,50,25 整合MF后预测时层数太深？

    cf_unit_nums=[100,50] # CF中 UV先整合再跟pair整合

    sim_feature_size = 8
    DHSR_layers1 = [32, 16, 8]
    DHSR_layers2 = [32, 16, 8]

    NCF_layers = [64, 32, 16, 8]
    NCF_reg_layers = [0.01, 0.01, 0.01, 0.01]
    NCF_reg_mf = 0.01

    num_negatives = 6 # 6 正常是6，为加速调试，设为2
    split_mannner = 'cold_start'  # 'cold_start' 冷启动问题研究 'left_one_spilt'一般问题最佳,'left_one_spilt','mashup'
    train_ratio = 0.7
    candidates_manner = 'all'  # 'num' 'ratio'  'all'
    candidates_num = 50#100
    candidates_ratio = 99
    s = ''  # 名称中的取值字段
    if candidates_manner == 'ratio':
        s = candidates_ratio
    elif candidates_manner == 'num':
        s = candidates_num

    # 划分数据集并获得样本***
    ds = split_dataset(data_dir, split_mannner, train_ratio, num_negatives)
    data_name = '{}_{}_{}_{}_{}'.format(split_mannner, train_ratio, candidates_manner, s, num_negatives)
    ds.split_dataset()

    train_mashup_id_list, train_api_id_list, train_labels, test_mashup_id_list, test_api_id_list, grounds = ds.get_model_instances(
        candidates_manner, candidates_num=candidates_num)
    # train_UI_matrix=transfer_UI_matrix(zip(train_mashup_id_list,train_api_id_list)) # 使用原始的UI还是复采样之后的UI?
    train_UI_matrix = transfer_UI_matrix (ds.train_mashup_api_list).get_transfered_UI('matrix')  # 供CF中的0-1向量使用

    # 为了使用预训练的特征提取器 提取mashup的text/tag feature
    feature_train_mashup_ids = list(np.unique(train_mashup_id_list)) # 训练用mashup的升序排列
    feature_train_api_ids = [0] * len(feature_train_mashup_ids)

    # best_epoch_path 确定哪种模型的哪次epoch效果最优；
    # 据此加载text_tag_para_path中的模型参数，再读取text_features_path中的feature设置CF模型中的feature参数
    best_epoch_path=os.path.join(history_result_path,'best_epoch_{}')   # .format(para_mode)
    model_para_path=os.path.join(history_result_path,'text_tag_CF_model_{}_weights_{}.h5')# .format(para_mode,best_epoch)
    text_features_path=os.path.join(features_result_path,'text_feature_{}.txt')  # .format(para_mode)
    tag_features_path=os.path.join(features_result_path,'tag_feature_{}.txt')

    # cf/mf
    """
    mf_mode='pmf' # 'pmf','BPR','listRank','node2vec'
    num_feat=25
    # 获取mashup,api 的latent factor  user，item：内部索引   item需要记录id到index的映射  id从小到大的list
    u_factors_matrix,m_ids,i_factors_matrix, a_ids=get_UV(ds.result_path, mf_mode, ds.train_mashup_api_list)
    i_id2index={id:index for index,id in enumerate(a_ids)}
    """

    num_feat = 25

    @classmethod
    def set_MF_mode(cls,mf_mode,num_feat = 25):# 'pmf','BPR','listRank','Node2vec'
        # cf/mf
        cls.mf_mode=mf_mode
        cls.num_feat=num_feat
        # 获取mashup,api 的latent factor  user，item：内部索引   item需要记录id到index的映射  id从小到大的list
        cls.u_factors_matrix, cls.m_ids, cls.i_factors_matrix, cls.a_ids = get_UV (cls.ds.result_path, mf_mode, cls.ds.train_mashup_api_list)
        cls.i_id2index = {id: index for index, id in enumerate (cls.a_ids)}
        cls.m_id2index = {id: index for index, id in enumerate (cls.m_ids)}

    deep_co_fc_unit_nums=[1024,256,64,16]
    shadow_co_fc_unit_nums = [64, 16]


if __name__ == "__main__":
    mf_modes=['pmf','BPR','listRank','nmf','Node2vec']
    for mf_mode in mf_modes:
        Para.set_MF_mode (mf_mode)
        print (Para.u_factors_matrix[0])