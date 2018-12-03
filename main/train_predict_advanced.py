# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from main.helper import get_iniFeaturesAndParas

from main.para_setting import Para
from main.train_predict_methods import get_preTrained_text_tag_model, train_test_text_tag_CF_model
from recommend_models.baseline import Samanta, binary_keyword
from main.evalute import test_evalute
from recommend_models.recommend_Model import DHSR_model, NCF_model, DHSR_noMF
from recommend_models.text_only_model import gx_text_only_model,gx_text_only_MF_model
from recommend_models.text_tag_model import gx_text_tag_model, gx_text_tag_MF_model, gx_text_tag_CF_model, \
    gx_text_tag_only_MLP_model


def tst_para(model_name,text_extracter_modes,Category_types, tag_manners, merge_manners):
    def easy_call(recommend_model):
        model = recommend_model.get_model()
        csv_table_name = recommend_model.get_name()
        test_evalute(recommend_model, model, csv_table_name, Para.num_epochs, train=True)

    recommend_model=None
    if model_name == 'DHSR':
        recommend_model = DHSR_model(Para.data_dir, Para.mf_embedding_dim, Para.sim_feature_size, Para.DHSR_layers1,Para.DHSR_layers2)
        easy_call(recommend_model)
    elif model_name == 'DHSR_noMF':
        recommend_model = DHSR_noMF(Para.data_dir, Para.mf_embedding_dim, Para.sim_feature_size, Para.DHSR_layers1, Para.DHSR_layers2)
        easy_call(recommend_model)
    elif model_name == 'NCF':
        recommend_model = NCF_model(Para.data_dir, Para.mf_embedding_dim, Para.NCF_layers, Para.NCF_reg_layers, Para.NCF_reg_mf)
        easy_call(recommend_model)
    else:
        for text_extracter_mode in text_extracter_modes:
            for Category_type in Category_types: # 三种不同搭配方式
                for merge_manner in merge_manners:
                    for tag_manner in tag_manners:
                        print(Category_type,tag_manner,merge_manner)
                        GX_base_parameters= Para.data_dir, Para.remove_punctuation, Para.embedding_name, Para.embedding_dim, text_extracter_mode, Para.inception_channels, \
                                            Para.inception_pooling, Para.inception_fc_unit_nums, Para.content_fc_unit_nums
                        # 获得模型
                        recommend_model=None
                        if model_name == 'text_only':
                            recommend_model = gx_text_only_model(*GX_base_parameters)  # Build model
                        elif model_name == 'text_only_MF':
                            recommend_model = gx_text_only_MF_model(*GX_base_parameters, Para.mf_embedding_dim, Para.mf_fc_unit_nums,
                                                                    Para.predict_fc_unit_nums)
                        elif model_name == 'text_tag':
                            recommend_model = gx_text_tag_model(*GX_base_parameters, Category_type, tag_manner, merge_manner,
                                                                Para.text_fc_unit_nums, Para.tag_fc_unit_nums)
                        elif model_name == 'text_tag_MF':
                            recommend_model = gx_text_tag_MF_model(*GX_base_parameters, Category_type, tag_manner, merge_manner,
                                                                   Para.mf_embedding_dim, Para.mf_fc_unit_nums, Para.predict_fc_unit_nums,
                                                                   Para.text_fc_unit_nums, Para.tag_fc_unit_nums)
                        else:
                            raise  ValueError('input wrong model name!')

                        easy_call(recommend_model)


def tst_inceptionCF(text_extracter_mode,Category_type, tag_manner, merge_manner, para_mode,CF_self_1st_merge,text_weight,if_co,if_pop):
    GX_base_parameters = Para.data_dir, Para.remove_punctuation, Para.embedding_name, Para.embedding_dim, text_extracter_mode, Para.inception_channels, \
                         Para.inception_pooling, Para.inception_fc_unit_nums, Para.content_fc_unit_nums

    max_ks = [5, 10, 20, 30, 40, 50]  # ***

    pmf_01='01'

    # 获得text_tag_model
    text_tag_recommend_model = gx_text_tag_model(*GX_base_parameters, Category_type, tag_manner, merge_manner,
                                                 Para.text_fc_unit_nums, Para.tag_fc_unit_nums)
    text_tag_model = text_tag_recommend_model.get_model()

    if para_mode == 'Samanta':
        Samanta(text_tag_recommend_model, max_ks[-1])
    if para_mode == 'binary_keyword':
        binary_keyword(text_tag_recommend_model)
    else:
        # 预训练 text_tag_model，得到mashup的features；和预训练的模型参数；是否重新训练
        ini_features_array, text_tag_model=get_preTrained_text_tag_model(text_tag_recommend_model,text_tag_model,train_new=False)
        print('pre_train text_tag_model, done!')

        if para_mode == 'MLP_only':
            test_times=1
            for i in range(test_times):
                # trian_MLP_only(text_tag_recommend_model, text_tag_model, ini_features_array, max_ks[-1], predict_fc_unit_nums)
                text_tag_MLP_only_recommend_model=gx_text_tag_only_MLP_model(*GX_base_parameters, Category_type, tag_manner, merge_manner,
                                                                             Para.mf_embedding_dim, Para.mf_fc_unit_nums,
                                                                             Para.u_factors_matrix,Para.i_factors_matrix,Para.m_id2index,Para.i_id2index,ini_features_array, max_ks,Para.num_feat,max_ks[-1],CF_self_1st_merge,Para.cf_unit_nums,text_weight,
                                                                             Para.predict_fc_unit_nums, Para.text_fc_unit_nums,
                                                                             Para.tag_fc_unit_nums,if_co,if_pop,Para.shadow_co_fc_unit_nums if if_co==3 else Para.deep_co_fc_unit_nums
                                                                             )
                text_tag_MLP_only_recommend_model.initialize(text_tag_recommend_model,text_tag_model,Para.train_mashup_id_list,
                                                             Para.train_api_id_list,Para.test_mashup_id_list,Para.test_api_id_list,Para.feature_train_mashup_ids)

                text_tag_MLP_only_model=text_tag_MLP_only_recommend_model.get_model()
                # 是否重新训练上层的MLP：是
                test_evalute(text_tag_MLP_only_recommend_model, text_tag_MLP_only_model, text_tag_MLP_only_recommend_model.get_name()+'_'+Para.mf_mode, Para.num_epochs, train=True)

        # 1,12,13,123三种模式，单独训练MLP，训练全部参数等等... 需要调用train_test_text_tag_CF_model() 在text_tag基础上训练模型
        else:
            if para_mode != '1':
                # 先根据text-tag的结果更新feature
                text_tag_CF_recommend_model = gx_text_tag_CF_model(*GX_base_parameters, Category_type, tag_manner, merge_manner,
                                                                   Para.mf_embedding_dim, Para.mf_fc_unit_nums,
                                                                   Para.pmf_01,Para.train_UI_matrix,Para.u_factors_matrix,Para.i_factors_matrix,Para.a_ids,ini_features_array, max_ks,
                                                                   Para.predict_fc_unit_nums, Para.text_fc_unit_nums, Para.tag_fc_unit_nums)

                text_tag_CF_model=None
                # 根据效果最优的text_tag模型的参数搭建新的text_tag_CF，默认将使用共同的embedding，text_feature,tag_feature提取器参数
                if para_mode == '12' or para_mode == '13':
                    text_tag_CF_model=text_tag_CF_recommend_model.get_model(text_tag_model)
                    print('text_tag_CF_model based on text_tag_model, build done!')
                elif para_mode == '123': # 得到参数已部分训练的model和对应的feature
                    # text_tag_CF_model = text_tag_CF_recommend_model.get_model() # 有问题***
                    text_tag_CF_model = text_tag_CF_recommend_model.get_model(text_tag_model)
                    ini_features_array, text_tag_CF_model=get_iniFeaturesAndParas(text_tag_CF_model, '12')
                    text_tag_CF_recommend_model.update_features(ini_features_array)

                train_new=True # 多次训练时覆盖即可，123调用12时设为False
                if para_mode[-1]=='2':
                    # 先预训练上层MLP
                    train_test_text_tag_CF_model(text_tag_CF_recommend_model, text_tag_CF_model, False,para_mode,text_tag_model,train_new=train_new)
                    print('fine_tuning, train MLP, done!')
                elif para_mode[-1]=='3':
                    # 全部参数训练
                    train_test_text_tag_CF_model(text_tag_CF_recommend_model, text_tag_CF_model,True,para_mode,train_new=train_new)
                    print('train all paras, done!')


if __name__ == '__main__':
    model_names=['DHSR_noMF'] # 'DHSR','DHSR_noMF','NCF'  'text_only', 'text_only_MF','text_tag','text_tag_MF','text_tag_CF'
    text_extracter_modes=['inception'] # 'inception','LSTM','textCNN'
    Category_types=['all'] # ,'first'
    tag_manners=['old_average'] #'old_average'
    merge_manners=['direct_merge'] #'final_merge','direct_merge'
    times=1

    """
    for i in range(times):
        for model_name in model_names:
            tst_para(model_name,text_extracter_modes,Category_types, tag_manners, merge_manners)
    """
    CF_self_1st_merge=True
    text_weights=[0.0]#0.0+0.02*i for i in range(4)
    mf_modes=['Node2vec'] # 'listRank','BPR'
    if_cos=[0,1,2,3]
    if_pops=[False,False,False,False]
    baseline_model='binary_keyword' #'Samanta'
    model_name= 'MLP_only'

    for mf_mode in mf_modes:
        Para.set_MF_mode (mf_mode)
        print('test {}'.format(mf_mode))
        for text_weight in text_weights:
                for i in range(len(if_cos)):
                    tst_inceptionCF(text_extracter_modes[0],Category_types[0], tag_manners[0], merge_manners[0],model_name,CF_self_1st_merge,text_weight,if_cos[i],if_pops[i]) #

