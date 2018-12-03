# encoding:utf8
import csv

import numpy as np
from numpy.linalg import linalg


def write_mashup_api_pair(mashup_api_pairs, data_path, manner):
    # 存储 mashup api 关系对
    with open(data_path, 'w+') as f:
        if manner == 'list':
            for mashup_id, api_id in mashup_api_pairs:
                f.write("{}\t{}\n".format(mashup_id, api_id))
        elif manner == 'dict':
            for mashup_id, api_ids in mashup_api_pairs.items():
                for api_id in api_ids:
                    f.write("{}\t{}\n".format(mashup_id, api_id))
        else:
            pass


def list2dict(train):
    """
    将（UID，iID）形式的数据集转化为dict  deepcopy
    :param train:
    :return:
    """
    a_dict = {}
    for mashup_id, api_id in train:
        if mashup_id not in a_dict.keys():
            a_dict[mashup_id] = set()
        a_dict[mashup_id].add(api_id)
    return a_dict


def dict2list(train):
    _list=[]
    for mashup,apis in train:
        for api in apis:
            _list.append(mashup,api)
    return _list

def cos_sim(A, B):
    if isinstance(A, list):
        A = np.array(A)
        B = np.array(B)
    num = float((A * B).sum())  # 若为列向量则 A.T  * B
    denom = linalg.norm(A) * linalg.norm(B)
    cos = num / denom  # 余弦值
    return cos

def Euclid_sim():
    pass

def transform_dict(a_dict):
    new_dict={}
    for key,value in a_dict.items():
        new_dict[value]=key
    return new_dict


def get_id2index(doc_path):  # librc处理后的id2index文件
    id2index = {}
    reader = csv.DictReader(open(doc_path, 'r'))  # r
    for row in reader:
        id2index[int(row['id'])] = int(row['index'])
    return id2index


def save_id2index(id2index,doc_path):
    with open(doc_path,'w') as f:
        f.write('id,index\n')
        for id,index in id2index.items():
            f.write('{},{}\n'.format(id,index))


if __name__=='__main__':
    a=[1,0]
    b=[-1,0]
    print(cos_sim(a,b))