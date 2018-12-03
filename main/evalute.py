import heapq
import os
import sys
from gensim.corpora import Dictionary
from process_text.processing_data import process_data

sys.path.append("..")
from keras.optimizers import Adam
import numpy as np
from prettytable import PrettyTable

from helpers.evaluator import evaluate
from main.para_setting import Para
import pickle


def test_evalute(recommend_model, model, csv_table_name, num_epochs, train=True,):
    """
    对test_mashup_id_list，test_api_id_list进行测试
    :param para类对象
    :param recommend_model: 可以是text_tag,CF,MF,only_MLP
    :param model:
    :param csv_table_name: 区分该模型的，写在结果中的名字
    :param train:是否先训练
    :return:
    """
    train_instances_tuple = None
    if train:
        train_instances_tuple = recommend_model.get_instances(Para.train_mashup_id_list, Para.train_api_id_list)
    else:
        num_epochs = 1

    evaluate_results=[] # 各个epoch的结果
    for epoch in range(num_epochs):
        predictions=[] # 分别测试哪种跟pop结合的方式好
        candidate_ids_list = []
        if train:
            if epoch==0:
                model.compile(optimizer=Adam(lr=Para.learning_rate), loss='binary_crossentropy')
            # Training
            print('Epoch {}'.format(epoch))
            model.fit([*train_instances_tuple], np.array(Para.train_labels), batch_size=Para.small_batch_size, epochs=1,
                      verbose=0,
                      shuffle=True)
            print('model train,done!')

        test_mashup_num = len(Para.test_mashup_id_list)
        csv_table_name=Para.data_name +csv_table_name+'\n'  if epoch==0 else ''
        csv_table_name +='epoch{}\n'.format(epoch)
        for i in range(test_mashup_num):
            candidate_ids = Para.test_api_id_list[i]
            candidate_ids_list.append(candidate_ids)

            test_batch_size = 50
            prediction=[]  # 每个mashup的全部预测结果
            # 因为test mashup太多，需要分batch预测
            test_api_num=len(candidate_ids)
            num = test_api_num // test_batch_size
            yushu=test_api_num % test_batch_size
            if yushu!=0:
                num+=1
            for j in range(num):
                stop_index = test_api_num if (yushu!=0 and j==num-1) else (j+1)*test_batch_size

                batch_api_ids=candidate_ids[j * test_batch_size:stop_index]
                batch_instances_tuple = recommend_model.get_instances(Para.test_mashup_id_list[i][j * test_batch_size:stop_index], batch_api_ids)  # 多态

                batch_prediction = list(model.predict([*batch_instances_tuple], verbose=0)) # 该批次的api评分
                prediction+=batch_prediction

            predictions.append(list(prediction))  # array 对象转化为list
        print('test,done!')

        evaluate_result = evalute(candidate_ids_list, predictions, Para.grounds, Para.topKs)  # 评价
        evaluate_results.append(evaluate_result)
        summary(Para.evaluate_path, csv_table_name, evaluate_result, Para.topKs)  # 记录 pop[0]
        recommend_model.save_sth()

        # 存储预测评分文件
        with open(os.path.join (Para.data_dir, 'model_predictions_{}.dat'.format(epoch)),'wb') as f:
            pickle.dump ((Para.test_mashup_id_list, Para.test_api_id_list, predictions),f) # 文件名太长？？？

        # 将模型评分与pop结合的各种尝试
        add_pop_predictions(recommend_model, csv_table_name,epoch)

    return evaluate_results


# 用于读取存储的预测结果，再跟pop值结合重新评价
def add_pop_predictions(recommend_model, csv_table_name,epoch, pop_mode='sigmoid', a_pop_ratio=0.0):
    test_mashup_id_list, test_api_id_list, predictions =None,None,None
    with open (os.path.join (Para.data_dir, 'model_predictions_{}.dat'.format (epoch)), 'rb') as f:
        test_mashup_id_list, test_api_id_list, predictions=pickle.load(f)

    api_id2covec, api_id2pop = recommend_model.pd.get_api_co_vecs (pop_mode=pop_mode)

    # 乘积
    predictions_pop=[]
    for m_index in range(len(predictions)):
        a_mashup_predictions=predictions[m_index]
        temp_preditions=[]
        for a_index in range(len(a_mashup_predictions)):
            a_prediction=a_mashup_predictions[a_index]
            api_id=test_api_id_list[m_index][a_index]
            temp_preditions.append(api_id2pop[api_id]*a_prediction)
        predictions_pop.append(temp_preditions)
    evaluate_result_linear_sum = evalute (test_api_id_list, predictions_pop, Para.grounds, Para.topKs)  # 评价
    summary (Para.evaluate_path, pop_mode+'_pop_prod\n' + csv_table_name, evaluate_result_linear_sum, Para.topKs)

    # 线性加权求和
    pop_ratios=[0.2+0.2*i for i in range(5)]
    for pop_ratio in pop_ratios:
        predictions_pop_linear = []
        for m_index in range(len (predictions)):
            a_mashup_predictions = predictions[m_index]
            temp_preditions = []
            for a_index in range(len (a_mashup_predictions)):
                a_prediction = a_mashup_predictions[a_index]
                api_id = test_api_id_list[m_index][a_index]
                temp_preditions.append ((1 - pop_ratio) * a_prediction + pop_ratio *api_id2pop[api_id])
            predictions_pop_linear.append (temp_preditions)

        evaluate_result_linear_sum = evalute (test_api_id_list, predictions_pop_linear, Para.grounds, Para.topKs)  # 评价
        summary (Para.evaluate_path, pop_mode+'_pop_{}\n'.format(pop_ratio) + csv_table_name, evaluate_result_linear_sum, Para.topKs)

    predictions_pop_last=[]
    for m_index in range(len(predictions)):
        # 首先根据score选出候选
        score_mapping = [pair for pair in zip(test_api_id_list[m_index], predictions[m_index])]
        max_k_pairs = heapq.nlargest(100, score_mapping, key=lambda x: x[1]) # 根据score选取top100*
        max_k_candidates, _ = zip(*max_k_pairs)
        # 然后仅根据pop rank
        temp_preditions = [api_id2pop[api_id] if api_id in max_k_candidates else -1 for api_id in test_api_id_list[m_index]]
        predictions_pop_last.append(temp_preditions)

    evaluate_result_linear_sum = evalute (test_api_id_list, predictions_pop_last, Para.grounds, Para.topKs)  # 评价
    summary (Para.evaluate_path, pop_mode+'_pop_last\n' + csv_table_name, evaluate_result_linear_sum, Para.topKs)


def evalute(candidate_ids_list, predictions, grounds, topKs):
    """
    :param candidate_ids_list: 用于测试的api id [[,]...] 2d
    :param predictions: 对应的该对的预测评分 2d
    :param grounds: 实际的调用api id 2d
    :param topKs:  哪些topK
    :return:
    """
    max_k = topKs[-1]
    mashup_num = len(candidate_ids_list)
    result = np.zeros((mashup_num, len(topKs), 5))

    for index in range(mashup_num):  # 单个mashup评价
        score_mapping = [pair for pair in zip(candidate_ids_list[index], predictions[index])]
        max_k_pairs = heapq.nlargest(max_k, score_mapping, key=lambda x: x[1]) # 根据score选取top50
        max_k_candidates, _ = zip(*max_k_pairs)

        for k_idx, k in enumerate(topKs):  # 某个topK
            result[index, k_idx, :] = evaluate(max_k_candidates, grounds[index], k)  # 评价得到五个指标，K对NDCG等有用

    return np.average (result, axis=0)


def summary(evaluate_path, csv_table_name, evaluate_result, topKs, use_table=True, stream=sys.stdout):
    assert len(topKs) == len(evaluate_result)
    # console 打印 结果
    table = PrettyTable("TopK Precision Recall F1 NDCG MAP".split())
    for k_idx, topK in enumerate(topKs):
        table.add_row((topK, *("{:.4f}".format(val) for val in evaluate_result[k_idx])))
    stream.write(str(table))
    stream.write("\n")

    # csv 中保存结果
    csv_table = csv_table_name + "TopK,Precision,Recall,F1,NDCG,MAP\n" # 错误地调换了MAP和NDCG的顺序
    for k_idx, topK in enumerate(topKs):
        csv_table += "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(topK, *evaluate_result[k_idx])
    with open(evaluate_path, 'a+') as f1:
        f1.write(csv_table)

    return 0