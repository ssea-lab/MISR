# -*- coding:utf-8 -*-
from process_text.processing_data import process_data, get_mashup_api_field
import random
import os
import numpy as np
import pickle
from helpers.util import list2dict


class split_dataset(object):
    """
    老的仅提供按mashup划分的类，不再使用
    """
    def __init__(self, base_dir, split_mannner, train_ratio, num_negatives=6):  # ,valid_ratio
        """
        交叉验证？？？mashup划分时容易，但是按关系对比例划分时很难
        :param base_dir: 存取文件路径
        :param split_mannner: 受划分方式的影响：按mashup划分还是划分一定比例的关系对
        :param train_ratio: 训练集比例：按mashup划分时 是mashup比例；按关系对划分时是关系对比例
        valid_ratio:好像不怎么需要
        """

        self.base_dir = base_dir
        self.raw_data = process_data(self.base_dir)  # 未划分的数据集对象
        self.split_mannner = split_mannner
        self.train_radio = train_ratio
        # self.valid_ratio = valid_ratio
        self.num_negatives = num_negatives

    def split_dataset(self):
        if self.split_mannner=='mashup':
            return self.split_dataset_by_mashup()
        elif self.split_mannner == 'ratio':
            return self.split_dataset_by_ratio()
        else:
            raise ValueError('Wrong split_mannner!')

    def split_dataset_by_ratio(self, num_negatives=6):
        """
        根据比例划分
        :param num_negatives:
        :return:
        """
        return

    def split_dataset_by_mashup(self):
        """
        划分和选择负例作为一个整体，因为数据一起被使用
        :param num_negatives:
        :return:
        """
        result_path = self.base_dir + r'\mashup_split'  # 暂时定为 /mashup_split/train_set，需要改动***加入用户参数  K-folds
        if not os.path.exists(result_path + r'\train_set'):
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            mashup_ids = list(self.raw_data.get_mashup_api_index2name('mashup', index2name=True).keys())
            random.shuffle(mashup_ids)

            api_ids = set(self.raw_data.get_mashup_api_index2name('api', index2name=True).keys())
            mashup_api_dict = self.raw_data.get_mashup_api_pair('dict')
            mid = int(self.train_radio * len(mashup_ids))
            train_mashups = mashup_ids[:mid]
            test_mashups = mashup_ids[mid:]

            train_set = []  # (mashup,api)list
            train_labels = []  # 正负，而test只包含正例
            test_set = []

            for train_mashup_id in train_mashups:
                train_apis = mashup_api_dict[train_mashup_id]
                for api_id in train_apis:  # 所有
                    train_set.append((train_mashup_id, api_id))
                    train_labels.append(1)

                unobserved_list = list(api_ids - train_apis)
                random.shuffle(unobserved_list)
                for api_id in unobserved_list[:len(train_apis) * self.num_negatives]:
                    train_set.append((train_mashup_id, api_id))
                    train_labels.append(0)

            # 每个test mashup随机选一个正例
            for test_mashup_id in test_mashups:
                test_apis = mashup_api_dict[test_mashup_id]

                unobserved_list = list(api_ids - test_apis)
                random.shuffle(unobserved_list)
                for api_id in unobserved_list[:self.num_negatives]:  # 每个test项目只有一个用于正例
                    train_set.append((test_mashup_id, api_id))
                    train_labels.append(0)

                random_api_id = test_apis.pop()  # 随机一个正例作为训练集
                train_set.append((test_mashup_id, random_api_id))
                train_labels.append(1)

                # test
                for api_id in test_apis:
                    test_set.append((test_mashup_id, api_id))

            test_labels = np.zeros(len(test_set)) + 1

            """
            mid = int(self.valid_ratio*len(train_set))  # 验证集按关系对比例划分,占训练集的比例
            valid_set = train_set[:mid]
            train_set = train_set[mid:]
            return train_set,valid_set,test_set
            """

            self.save_spilt_result(result_path, train_set, train_labels, test_set, test_labels)
            return train_set, train_labels, test_set, test_labels
        else:
            return self.read_spilt_result(result_path)

    # 需要使用相同的划分数据集，所以一次随机划分之后应当保存
    def save_spilt_result(self, path, train_set, train_labels, test_set, test_labels):
        """
        应该根据划分方式和0.1.2确定保存路径***未完成
        """
        with open(os.path.join(path, 'train_set'), 'wb') as file1:  # 存储info
            pickle.dump(train_set, file1)
        with open(os.path.join(path, 'train_labels'), 'wb') as file2:  # 存储info
            pickle.dump(train_labels, file2)
        with open(os.path.join(path, 'test_set'), 'wb') as file3:  # 存储info
            pickle.dump(test_set, file3)
        with open(os.path.join(path, 'test_labels'), 'wb') as file4:  # 存储info
            pickle.dump(test_labels, file4)

    def read_spilt_result(self, path):
        """
        应该根据划分方式和0.1.2确定保存路径***未完成
        """
        with open(os.path.join(path, 'train_set'), 'rb') as file1:  # 存储info
            train_set = pickle.load(file1)
        with open(os.path.join(path, 'train_labels'), 'rb') as file2:  # 存储info
            train_labels = pickle.load(file2)
        with open(os.path.join(path, 'test_set'), 'rb') as file3:  # 存储info
            test_set = pickle.load(file3)
        with open(os.path.join(path, 'test_labels'), 'rb') as file4:  # 存储info
            test_labels = pickle.load(file4)
        return train_set, train_labels, test_set, test_labels

    def get_texts_by_pair(self, train_test):
        """ 示例
        :param train_test: 根据train/test set，返回对应的文本对
        :return:
        """
        mashup_id2info = self.raw_data.get_mashup_api_id2info('mashup')
        api_id2info = self.raw_data.get_mashup_api_id2info('api')
        mashup_id_list, api_id_list = zip(*train_test)
        mashup_text_input = [get_mashup_api_field(mashup_id2info, mashup_id, 'final_description') for mashup_id in
                             mashup_id_list]  # 可能为[]！！！
        api_text_input = [get_mashup_api_field(api_id2info, api_id, 'final_description') for api_id in api_id_list]
        return mashup_text_input, api_text_input

    def get_model_instances(self, train_set, train_labels, test_set, test_labels):
        """
        为model准备输入输出，此时不需要为每个id指定一个文本，这部分在最后做，否则太占空间没有意义
        :return:
        """
        """
        mashup_id2info = self.pd.get_mashup_api_id2info('mashup')
        api_id2info = self.pd.get_mashup_api_id2info('api')
        """
        mashup_num = len(self.raw_data.get_mashup_api_id2info('mashup'))
        api_num = len(self.raw_data.get_mashup_api_id2info('api'))

        # train 实例，list形式
        train_mashup_id_list, train_api_id_list = zip(*train_set)
        """
        train_mashup_text_input = [get_mashup_api_field(mashup_id2info, mashup_id, 'final_description') for mashup_id in train_mashup_id_list]  # 可能为[]！
        train_api_text_input = [get_mashup_api_field(api_id2info, api_id, 'final_description') for api_id in train_api_id_list]
        """

        # test实例，为了测试方便，以mashup为单位整理
        ground_set = list2dict(test_set)  # 实际选择
        train_dict = list2dict(train_set)  # train 的 dict形式
        test_mashup_id_list, test_api_id_list = [], [] #  test_mashup_text_input, test_api_text_input, [], []  # 2维数组.每一行batch处理

        all_api_ids = {api_id for api_id in range(api_num)}
        for mashup_id in range(mashup_num):
            if mashup_id in ground_set.keys():  # 按mashup划分时仅考虑test mashup 按比例划分时都要考虑***
                test_api_ids = list(all_api_ids - train_dict.get(mashup_id))  # 所有减去用于正负例的train，剩下的包括测试和unobserved
                test_api_num = len(test_api_ids)

                test_mashup_id_list.append([mashup_id] * test_api_num)
                """
                mashup_text = get_mashup_api_field(mashup_id2info, mashup_id, 'final_description')
                test_mashup_text_input.append([mashup_text] * test_api_num)
                """
                test_api_id_list.append(test_api_ids)
                # test_api_text_input.append([get_mashup_api_field(api_id2info, api_id, 'final_description') for api_id in test_api_ids])

        grounds = []
        for test_mashup_id, test_api_ids in ground_set.items():  # 只保留test_mashup_id_list的测试集当作ground
            grounds.append(list(test_api_ids))

        return train_mashup_id_list, train_api_id_list,  train_labels, \
               test_mashup_id_list, test_api_id_list, grounds #train_mashup_text_input, train_api_text_input,test_mashup_text_input, test_api_text_input


if __name__ == '__main__':
    ds = split_dataset(r'C:\Users\xiaot\Desktop\data', 'mashup', 0.7,6)
    split_result = ds.split_dataset()
    instance_result= ds.get_model_instances(*split_result)

    """
    print("划分结果")
    for r1 in result:
        print(len(r1))
        print(r1[0])
    """
    print("实例结果")
    for r2 in ds.get_model_instances(*split_result):
        print(np.array(r2).shape)
        print(r2[0])

