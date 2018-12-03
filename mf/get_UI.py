import os
import sys
sys.path.append("..")
import numpy as np

from helpers.util import get_id2index, save_id2index,write_mashup_api_pair
from mf.Node2Vec import  call_node2vec, args


def get_UV(data_path, mode, train_mashup_api_list):
    """

    :param data_path: 对哪种划分
    :param mode: 使用哪种矩阵分解方法
    :param train_mashup_api_list: 训练集
    :return:
    """
    MF_path=os.path.join(data_path,'U_V',mode)  # eg:C:\Users\xiaot\Desktop\MF+CNN\GX\data\split_data\cold_start\U_V\pmf\

    if not os.path.exists(MF_path):
        os.mkdir(MF_path)
        print(MF_path+' not exits,and created one!')
    m_ids,a_ids=zip(*train_mashup_api_list)
    m_ids=np.unique(m_ids)
    a_ids=np.unique(a_ids)

    m_embeddings, a_embeddings =None,None
    if mode=='Node2vec':
        m_embeddings, a_embeddings=get_UV_from_Node2vec(MF_path,train_mashup_api_list)
    else:
        m_embeddings=get_UV_from_librec(MF_path, "mashup", m_ids)
        a_embeddings=get_UV_from_librec(MF_path, "api", a_ids)
    return m_embeddings,m_ids,a_embeddings,a_ids


def get_UV_from_librec(MF_path, user_or_item, ordered_ids):
    """
    返回从librec得到的结果，按照id大小排列
    :param MF_path:
    :param user_or_item:
    :param ordered_ids: 一般是按照mashup，api的id从大到小排列的
    :return:
    """
    if user_or_item=="mashup":
        id2index_path=MF_path + "/userIdToIndex.csv"
        matrix_path=MF_path+"/U.txt"
    elif user_or_item == "api":
        id2index_path = MF_path + "/itemIdToIndex.csv"
        matrix_path = MF_path + "/V.txt"

    matrix=np.loadtxt(matrix_path)
    id2index=get_id2index(id2index_path)
    ordered_numpy=np.array([matrix[id2index[id]] for id in ordered_ids])
    return ordered_numpy


def prepare_data_for_Node2vec(a_args,train_mashup_api_list):
    """
    :param train_mashup_api_list: # 需传入内部索引？？？外部
    :return:
    """
    m_ids,a_ids=zip(*train_mashup_api_list)
    m_ids=np.unique(m_ids)
    a_ids=np.unique(a_ids)
    m_num=len(m_ids)

    # 对mashup和api的id进行统一
    m_id2index={m_ids[index]:index+1 for index in range(len(m_ids))}
    save_id2index (m_id2index, a_args.m_id_map_path)

    a_id2index = {a_ids[index]:m_num+index + 1 for index in range(len (a_ids))}
    save_id2index(a_id2index,a_args.a_id_map_path)

    pair=[]
    for m_id,a_id in train_mashup_api_list:
        pair.append((m_id2index[m_id],a_id2index[a_id])) # 内部索引
    write_mashup_api_pair(pair,a_args.input,'list')
    print('prepare_data_for_Node2vec,done!')


def get_UV_from_Node2vec(node2vec_path,train_mashup_api_list):
    """
    传入U-I,返回mashup和api的embedding矩阵,按照id大小排列
    :param node2vec_path:
    :param train_mashup_api_list:
    :return:
    """
    a_args= args(node2vec_path)
    if not os.path.exists(a_args.m_embedding):
        prepare_data_for_Node2vec(a_args,train_mashup_api_list)
        call_node2vec(a_args)

        m_ids,a_ids=zip(*train_mashup_api_list)
        m_ids = np.unique (m_ids)
        a_ids = np.unique (a_ids)

        index2embedding={}
        with open(a_args.output, 'r') as f:
            line=f.readline() # 第一行是信息
            line =f.readline()
            while(line):
                l=line.split(' ')
                index=int(l[0])
                embedding=[float(value) for value in l[1:]]
                index2embedding[index]=embedding

                line = f.readline ()

        m_id2index=get_id2index(a_args.m_id_map_path)
        a_id2index = get_id2index (a_args.a_id_map_path)

        m_embeddings=[]
        a_embeddings=[]
        # 按照id大小输出，外部使用
        for m_id in m_ids:
            m_embeddings.append(index2embedding[m_id2index[m_id]])
        for a_id in a_ids:
            a_embeddings.append(index2embedding[a_id2index[a_id]])

        np.savetxt(a_args.m_embedding, m_embeddings)
        np.savetxt (a_args.a_embedding, a_embeddings)
        return np.array(m_embeddings),np.array(a_embeddings)
    else:
        m_embeddings=np.loadtxt(a_args.m_embedding)
        a_embeddings=np.loadtxt (a_args.a_embedding)
        return m_embeddings,a_embeddings
    print ('get vecs of Node2vec,done!')

