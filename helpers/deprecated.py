
# 在更新在这里用作是text_tag_CF_model的全部参数时，需要在batch后更新其features
# 弃用***
from keras.callbacks import Callback
import keras.backend as K
import numpy as np

class update_features(Callback):
    def __init__(self,update_feature_instances,text_tag_CF_recommend_model):
        self.update_feature_instances=update_feature_instances
        self.text_tag_CF_recommend_model=text_tag_CF_recommend_model

    def on_batch_end(self, batch, logs={}):
        """
        self.model指代的是调用该callback的模型，在这里用作是text_tag_CF_model
        :return:
        """
        print('on_batch_end,update_features...')
        get_features = K.function(self.model.inputs,
                                          [self.model.get_layer('concatenate_1').input[0],
                                           self.model.get_layer('concatenate_1').input[2] ])
        text_features,tag_features = get_features(self.update_feature_instances)

        # 更新features变量（features是text_tag_CF_recommend_model的属性，唯一性，会影响到text_tag_CF_model?
        self.text_tag_CF_recommend_model.update_features(np.hstack((text_features,tag_features)))


def generater(train_instances_tuple,batch_size,train_labels=None):
    """

    :param train_instances_tuple:
    :param batch_size:
    :param train_labels:
    :return: tuple
    """
    if train_labels is not None:
        assert len(train_instances_tuple[0])==len(train_labels)

    size = len(train_instances_tuple[0])
    num = size // batch_size

    """
    if size % batch_size != 0:
        num += 1
    """

    for i in range(num):
        # stop_index=size if i==num-1 else (i+1)*batch_size
        stop_index = (i + 1) * batch_size
        if train_labels is not None:
            yield(
                [a_array[i*batch_size:stop_index] for a_array in train_instances_tuple],
                  train_labels[i*batch_size:stop_index]
                  )
        else:
            yield([a_array[i*batch_size:stop_index] for a_array in train_instances_tuple])