# -*- coding:utf-8 -*-
import os
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer,word_tokenize
from helpers import util
import pickle
import numpy as np
import math

doc_sparator = ' >>\t'
# punctuation =
# '[<>/\s+\.\!\/_,;:$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'
# 文本处理时去除特殊符号
english_stopwords = stopwords.words('english')  # 系统内置停用词
domain_stopwords = {}  # 专用语料停用词


class process_data(object):
    mashup_info_result = 'mashup.info'

    def __init__(self, base_path,process_new=False):  # 每个根目录一个对象
        self.base_dir = base_path
        self.processed = True if os.path.exists(os.path.join(base_path, process_data.mashup_info_result)) else False  # 是否计算过
        if not self.processed or process_new:
            self.process_raw_texts()

    def process_raw_texts(self):
        """
        .info包含所有mashup/api的信息  pickle封装dict
        name，id映射，以及mashup_api调用对则只有长度大于2的mashup参与统计  csv文件 读写接口
        """

        dirs = [os.path.join(self.base_dir, 'Mashup'), os.path.join(self.base_dir, 'API')]

        mashup_api = {}
        # 0 mashup 1 api 决定tag（类别）在文件中的位置
        for data_type, data_dir in enumerate(dirs):
            name2info = {}

            for doc_name in os.listdir(data_dir):
                name = ''
                a_dict = {}
                with open(os.path.join(data_dir, doc_name),encoding='utf-8') as f: #处理时所有mashup和api均写入
                    for line in f.readlines():
                        line = line.split(doc_sparator)
                        if len(line)<2 and 'Description' in a_dict.keys():#有时Description会跨行
                            a_dict['Description']=a_dict['Description']+line[0]
                            continue

                        Key=line[0]
                        Value=line[1]
                        if Key == 'Name':
                            name = '-'.join(Value.strip().lower().split()) # name 有空格，转化为跟related apis中相同的格式
                        elif Key in ['Primary Category','Secondary Categories','Categories','Description']: # 类别词和描述词要小写
                            Value = Value.lower().strip()
                            if Key in ['Primary Category','Secondary Categories','Categories']:# 类别词进一步处理，跟文本相同的list形式，方便操作
                                Value=  Value.split(', ') # 多个类别使用', '分隔
                            a_dict[Key]=Value
                        elif Key =="Related APIs":
                            a_dict[Key] = Value
                            if len(Value.split()) >= 2:  # 统计计数只考虑长度大于2的
                                mashup_api[name] = [api.strip().lower() for api in Value.split()] # Related api中的名称和 name段的经过相同的处理
                        else:
                            pass
                if a_dict.get('Description') is not None:
                    a_dict['final_description'] = NLP_tool(a_dict['Description'])
                name2info[name] = a_dict  # name到对应info的字典

            final_docname = 'mashup.info' if data_type == 0 else 'api.info'
            with open(os.path.join(self.base_dir,final_docname), 'wb') as file:  # 存储info
                pickle.dump(name2info, file)
        print("write text, done!")

        # 处理调用关系，得到mashup,api的name-id映射，以及id形式的关系对
        mashup_name2index = {}
        api_name2index = {}
        mashup_index = 0
        api_index = 0
        for mashup_name, api_names in mashup_api.items():
            if mashup_name not in mashup_name2index.keys():
                mashup_name2index[mashup_name] = mashup_index
                mashup_index += 1
                for api_name in api_names:
                    if api_name not in api_name2index.keys():
                        api_name2index[api_name] = api_index
                        api_index += 1

        with open(os.path.join(self.base_dir, 'mashup_name2index'), 'wb') as file:  # 存储name到id映射
            pickle.dump(mashup_name2index, file)
        with open(os.path.join(self.base_dir, 'api_name2index'), 'wb') as file:
            pickle.dump(api_name2index, file)

        """
        # 存储mashup/api  name到id的映射   不使用，太麻烦
        with open(os.root_path.join(self.data_dir, 'mashup_name2index.csv'), 'w+',encoding='utf-8') as f1: # 一些字符是utf-16的,16能表示这些字符但是不能写,换格，所以还是用8
            f1.write("{},{}\n".format('mashup_name', 'mashup_id'))
            for mashup_name, mashup_id in mashup_name2index.items():
                print(mashup_name)
                f1.write("{},{}\n".format(mashup_name, mashup_id))
        with open(os.root_path.join(self.data_dir, 'api_name2index.csv'), 'w+',encoding='utf-8') as f2:
            f2.write("{},{}\n".format('api_name', 'api_id'))
            for api_name, api_id in api_name2index.items():
                f2.write("{},{}\n".format(api_name, api_id))
        """
        print("write index2name,done!")
        print("Num of mashup:{},Num of api:{}!".format(len(mashup_name2index),len(api_name2index)))

        mashup_api_pairs = []  # mashup，api调用关系对，id 形式
        for mashup_name, api_names in mashup_api.items():
            for api_name in api_names:
                mashup_api_pairs.append((mashup_name2index[mashup_name], api_name2index[api_name]))

        # 存储 mashup api 关系对
        util.write_mashup_api_pair(
            mashup_api_pairs, os.path.join(self.base_dir, 'mashup_api'), 'list')
        print("write mashup_api_pair,done!")

    def get_mashup_api_info(self, mashup_or_api):
        """
        返回mashup/api 的名称到info(text/tag)的映射  return 1 dicts:  string->dict
        """
        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")
        else:
            final_docname = os.path.join(self.base_dir, mashup_or_api + '.info')  # 文件名为mashup/api

        with open(final_docname, 'rb') as file2:
            return pickle.load(file2)

    def get_mashup_api_index2name(self, mashup_or_api, index2name=True):
        # 返回mashup/api 的名称到index的映射  默认是id-name

        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")
        else:
            map_path = mashup_or_api + '_name2index' # +'.csv'

        a_map = {}
        name2index={}

        with open(os.path.join(self.base_dir, map_path), 'rb') as file2:
            name2index=pickle.load(file2)
        if index2name:
            for name,index in name2index.items():
                a_map[index]=name
        return a_map if index2name else name2index


        """
        #csv版本，不使用
        id_column = mashup_or_api + '_id'
        name_column = mashup_or_api + '_name'
        
        reader = csv.DictReader(open(os.root_path.join(self.data_dir, map_path), 'r',encoding='utf-8'))  # r
        for row in reader:
            if index2name:
                # eg: mashup_map [int(row['mashup_id'])]= row['mashup_name']
                a_map[int(row[id_column])] = row[name_column]
            else:
                a_map[row[name_column]] = int(row[id_column])
        """
        return a_map

    def get_mashup_api_id2info(self, mashup_or_api):
        # 返回由id直接得到info的dict  用在将关系对和对应的text输入模型
        # 问题，有的

        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")

        name2info = self.get_mashup_api_info(mashup_or_api)
        index2name = self.get_mashup_api_index2name(mashup_or_api)

        id2info = {}
        for id, name in index2name.items():
            info=name2info.get(name)
            id2info[id]=info if info is not None else {}#可能一个被mashsup调用的api可能没有相关信息，此时值为{}!进而使得不存在信息的api的.get()=None

        return id2info

    def get_mashup_api_pair(self, manner):
        """
        获取关系对：pair list:[(m,a1),(m,a2)]  or  dict{(m:{a1,a2})} key:set!!!
        para:
        manner: 'list' or 'dict'
        """
        if not (manner == 'list' or manner == 'dict'):
            raise ValueError("must input 'list' or 'dict' ")

        a_list = []
        a_dict = {}
        with open(os.path.join(self.base_dir, 'mashup_api'), 'r') as f:
            for line in f.readlines():
                if line is not None:
                    #print(line)
                    line = line.strip().split('\t')
                    m_id = int(line[0])
                    api_id = int(line[1])
                    if manner == 'list':
                        a_list.append((m_id, api_id))
                    if manner == 'dict':
                        if m_id not in a_dict:
                            a_dict[m_id] = set()
                        a_dict[m_id].add(api_id)

        return a_list if manner == 'list' else a_dict

    def get_api_co_vecs(self,pop_mode=''): # pop数值是否规约到0-1？？？ 要改动
        """
        返回每个api跟所有api的共现次数向量和每个api的popularity
        :return:
        """
        all_api_num=len(self.get_mashup_api_index2name('api'))
        api_co_vecs = np.zeros ((all_api_num,all_api_num),dtype='float32')
        api2pop=np.zeros((all_api_num,),dtype='float32')
        mashup_api_pair=self.get_mashup_api_pair('dict')
        for mashup,apis in mashup_api_pair.items():
            for api1 in apis:
                api2pop[api1]+=1.0
                for api2 in apis:
                    if api1!=api2:
                        api_co_vecs[api1][api2]+=1.0
        if pop_mode=='sigmoid':
            api2pop=[1.0/(1+pow(math.e,-1*pop)) for pop in api2pop]
        return api_co_vecs,api2pop

    def get_all_texts(self,Category_type='all'):
        """
        得到所有mashup api的description和category 按index排列
        :param data_dir:
        :return: 每个mashup返回的是整个字符串！！！信息不存在则为''
        """
        num_users = len(self.get_mashup_api_index2name('mashup'))
        num_items = len(self.get_mashup_api_index2name('api'))

        mashup_id2info = self.get_mashup_api_id2info('mashup')
        api_id2info = self.get_mashup_api_id2info('api')
        mashup_descriptions = [' '.join(get_mashup_api_field(mashup_id2info, mashup_id, 'final_description'))+' ' for mashup_id
                               in range(num_users)]
        api_descriptions = [' '.join(get_mashup_api_field(api_id2info, api_id, 'final_description'))+' ' for api_id in
                            range(num_items)]

        mashup_categories = [' '.join(get_mashup_api_allCategories('mashup', mashup_id2info, mashup_id,Category_type))+' ' for mashup_id in
                             range(num_users)]
        api_categories = [' '.join(get_mashup_api_allCategories('api', api_id2info, api_id,Category_type))+' ' for api_id in
                          range(num_items)]
        return mashup_descriptions,api_descriptions,mashup_categories,api_categories


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def get_mashup_api_field(id2info,id,field):
    """
    返回一个id的对应域的值
    :param id2info:
    :param id:
    :param field: final_des,各种类别   ['','']
    :return:['','']  当该id无信息或者无该域时，返回[]
    """
    info= id2info.get(id)
    if info is None or info.get(field) is None: # 短路
        return []
    else:
        return info.get(field)


def get_mashup_api_allCategories(mashup_or_api,id2info,id,Category_type='all'):
    """
    返回一个mashup api所有类别词
    :return:['','']  当该id无信息或者无该域时，返回[]
    """

    info= id2info.get(id)
    if mashup_or_api=='mashup':
        Categories= get_mashup_api_field(id2info,id,'Categories')
        return Categories
    elif mashup_or_api=='api':
        Primary_Category=get_mashup_api_field(id2info,id,'Primary Category')
        Secondary_Categories= get_mashup_api_field(id2info,id,'Secondary Categories')
        if Category_type=='all':
            Categories=Primary_Category+Secondary_Categories
        elif Category_type=='first':
            Categories = Primary_Category
        elif Category_type=='second':
            Categories = Secondary_Categories
        return Categories
    else:
        raise ValueError('wrong mashup_or_api!')


def NLP_tool(raw_description, SpellCheck=False):  # 功能需进一步确认！！！
    """
    返回每个文本预处理后的词列表:
    return [[],[]...]
    """

    """ 拼写检查
    d=None
    if SpellCheck:
        d = enchant.Dict("en_US")
    """

    # st = LancasterStemmer()  # 词干分析器

    words = []
    """ 
    line = re.sub(punctuaion, ' ', text)  # 不去标点，标点有一定含义
    words= line.split()
    """
    for sentence in tokenizer.tokenize(raw_description):  # 分句再分词
        #for word in WordPunctTokenizer().tokenize(sentence): #分词更严格，eg:top-rated会分开
        for word in word_tokenize(sentence):
            word=word.lower()
            if word not in english_stopwords and word not in domain_stopwords:  # 非停用词
                """
                if SpellCheck and not d.check(word):#拼写错误，使用第一个选择替换？还是直接代替？
                    word=d.suggest(word.lower())[0]
                """
                # word = st.stem(word)   词干化，词干在预训练中不存在怎么办? 所以暂不使用
                words.append(word)

    return words


def test_NLP(text):
    for sentence in tokenizer.tokenize(text):  # 分句再分词
        for word in WordPunctTokenizer().tokenize(sentence):
            print(word + "\n")

def test_utf():
    data_path = r'../mashup/%E2%96%B2hail'
    with open(data_path,encoding='utf-8') as f:
        print(f.readline())

if __name__ == '__main__':
    # test_NLP('i love you, New York.')

    test_data_dir = r'../test_data'
    real_data_dir= r'../data'
    pd = process_data(real_data_dir,True) #

    """
    for name, info in pd.get_mashup_api_info('mashup').items():
        print(name,info.get('final_description'))
    for name, info in pd.get_mashup_api_info('api').items():
        print(name, info.get('final_description'))
        print(name, info.get('Secondary Categories'))
    """

    """
    mashup_api_pairs = pd.get_mashup_api_pair('list')
    print(mashup_api_pairs)
    # 不存在信息的mashup/api的info获取
    name2info = pd.get_mashup_api_info('api')
    print(name2info)
    
    api_id2info=pd.get_mashup_api_id2info('api')
    for id, info in api_id2info.items():
        print(info.get('final_description'))
    """