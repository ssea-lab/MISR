import sys
sys.path.append("..")
from main.helper import get_iniFeaturesAndParas


import os
import numpy as np
from keras import Model
from keras.optimizers import Adam

from main.evalute import test_evalute, evalute
from main.para_setting import Para


def get_preTrained_text_tag_model(text_tag_recommend_model, text_tag_model, para_mode='1', train_new=False):
    if os.path.exists(Para.best_epoch_path.format(para_mode)) and not train_new:  # 利用求过的结果
        print('preTrained_text_tag_model, best text_feature, exists!')
        return get_iniFeaturesAndParas(text_tag_model, para_mode)
    else:
        text_tag_model.compile(optimizer=Adam(lr=Para.learning_rate), loss='binary_crossentropy')

        train_instances_tuple = text_tag_recommend_model.get_instances(Para.train_mashup_id_list, Para.train_api_id_list)
        feature_instances_tuple = text_tag_recommend_model.get_instances(Para.feature_train_mashup_ids,
                                                                         Para.feature_train_api_ids, True)  # 只包含mashup信息

        best_epoch = -1
        best_NDCG_5 = 0
        mashup_text_tag_features = []  # 存储各个epoch下的feature

        # 每轮迭代后均预测评价，存储该模型参数和对应的feature；
        # 最终可选择指标最优的模型，及其特征；模型参数用来初始化text_tag_CF模型中相同的层，特征则用来初始化
        print('pre-train text_tag_model:')
        for epoch in range(Para.num_epochs):
            # Training
            print('Epoch {}'.format(epoch))
            hist = text_tag_model.fit([*train_instances_tuple], np.array(Para.train_labels), batch_size=Para.small_batch_size,
                                      epochs=1, verbose=0,
                                      shuffle=True)  #
            print('text_tag_model train,done!')

            # 将各个epoch提取的mashup特征存储，用于初始化CF方法中每个mashup的feature
            """
            # 有问题，输入输出不在一个计算图上  可能是text_tag_model直接调用封装好的model造成的***
            # text_tag_middle_model=Model(inputs=text_tag_model.inputs,outputs=[text_tag_model.get_layer('text_feature_extracter').get_output_at(0), text_tag_model.get_layer('categories_feature_extracter').get_output_at(0)])

            # 可以，但是需要使用封装model的输入和输出;多层嵌入时不好用
            # text_feature_extracter 提取器和categories_feature_extracter 被分别当作一个层，名称是model_1和model_2
            text_tag_middle_model = Model(inputs=[text_tag_model.get_layer('model_1').get_input_at(0),text_tag_model.get_layer('model_2').get_input_at(0)],
                                          outputs=[text_tag_model.get_layer('model_1').get_output_at(0),
                                                   text_tag_model.get_layer('model_2').get_output_at(0)])
            """
            # 使用外部model的输入和输出
            text_tag_middle_model = Model(inputs=[text_tag_model.inputs[0], text_tag_model.inputs[2]],
                                          outputs=[text_tag_model.get_layer('concatenate_1').input[0],
                                                   text_tag_model.get_layer('concatenate_1').input[2]])

            print('text_tag_middle_model, build done!')
            mashup_text_tag_features.append(text_tag_middle_model.predict([*feature_instances_tuple], verbose=0))

            csv_table_name =  text_tag_recommend_model.get_name() if epoch == 0 else ''
            evaluate_result = test_evalute(text_tag_recommend_model, text_tag_model, csv_table_name, 1,
                                           train=False)

            text_tag_model.save_weights(Para.model_para_path.format(para_mode, epoch))  # 记录该epoch下的模型参数***

            if evaluate_result[0][3] > best_NDCG_5:  # 记录NDCG_5效果最好的模型下  提取的文本特征
                best_NDCG_5 = evaluate_result[0][3]
                best_epoch = epoch

        ini_mashup_text_feature = mashup_text_tag_features[best_epoch][0]  # 效果最好的epoch下的feature
        ini_mashup_tag_feature = mashup_text_tag_features[best_epoch][1]
        ini_features_array = np.hstack((ini_mashup_text_feature, ini_mashup_tag_feature))  # 整合的mashup的特征

        np.savetxt(Para.text_features_path.format(para_mode), ini_mashup_text_feature)
        np.savetxt(Para.tag_features_path.format(para_mode), ini_mashup_tag_feature)

        with open(Para.best_epoch_path.format(para_mode), 'w') as f:
            f.write(str(best_epoch))
        print('best epoch:{}'.format(best_epoch))

        # 返回效果最好的text_tag_CF_model
        text_tag_model.load_weights(Para.model_para_path.format(para_mode, best_epoch))

        return ini_features_array, text_tag_model


def train_test_text_tag_CF_model(text_tag_CF_recommend_model, text_tag_CF_model, trainable, para_mode,
                                 text_tag_model=None, train_new=True):
    """
    fine_tuning and update the paras of MLP
    :param text_tag_model:
    :param text_tag_CF_recommend_model:
    :param text_tag_CF_model:
    :param mode:固定Extracter与否，即是否更新全部参数  frozenExtracter_updateMLP(fine_tuning),update_text_tag_CF
    :param para_mode '12' '13' '123' pretrain text_tagmodel之后，只更新MLP，只更新全部，和先更新MLP再更新全部的缩写
    12，13只会被123调用 绝大多数多次测试12,13直接覆盖，训练新模型即可***
    :return:  MLP更新过的text_tag_CF_recommend_model
    """
    if os.path.exists(Para.best_epoch_path.format(para_mode)) and not train_new:  # 利用求过的结果
        print('text_tag_CF_model_{}, best text_feature, exists!'.format(para_mode))
        return get_iniFeaturesAndParas(text_tag_CF_model, para_mode)
    else:
        # 根据2/3指定文本部分layer是否固定
        if trainable:  # 只在12时 固定
            for layer in text_tag_CF_model.layers:
                if layer in text_tag_model.layers:
                    layer.trainable = trainable
                    print(layer.name)

        text_tag_CF_train_instances_tuple = text_tag_CF_recommend_model.get_instances(Para.train_mashup_id_list,
                                                                                      Para.train_api_id_list)

        # 在数据集上训练测试text_tag_CF_model
        text_tag_CF_model.compile(optimizer=Adam(lr=Para.learning_rate), loss='binary_crossentropy')

        best_epoch = -1
        best_NDCG_5 = 0

        # 名称引入mode,数字***
        para_path_part = r'/text_tag_CF_model_{}_weights_{}.h5'

        update_feature_text_tag_model = None
        text_features, tag_features = None, None

        feature_instances_tuple = text_tag_CF_recommend_model.get_instances(Para.feature_train_mashup_ids,
                                                                            Para.feature_train_api_ids)
        user_input_samples = np.expand_dims(feature_instances_tuple[0], axis=1)  # (2936,)->(2936,1)
        item_input_samples = np.expand_dims(feature_instances_tuple[1], axis=1)  # (2936,)->(2936,1)
        update_feature_instances = [user_input_samples, item_input_samples, *feature_instances_tuple[2:]]
        """
        # callback 中更新用的样本
        callback_list=[update_features(update_feature_instances,text_tag_CF_recommend_model)] if mode=='update_text_tag_CF' else []

        # k.function()
        get_features = K.function(text_tag_CF_model.inputs,
                                  [text_tag_CF_model.get_layer('concatenate_1').input[0],
                                   text_tag_CF_model.get_layer('concatenate_1').input[2]])
        """

        for epoch in range(Para.num_epochs):  # 每轮迭代后均预测评价，可选择最终指标最优的模型
            print('Epoch {}'.format(epoch))
            print(len(text_tag_CF_train_instances_tuple[0]))

            # Training
            """
            # generator+callback 版本内存溢出
            hist = text_tag_CF_model.fit_generator(generater(text_tag_CF_train_instances_tuple, batch_size, train_labels),len(text_tag_CF_train_instances_tuple[0]) // batch_size ,epochs=1, verbose=1,callbacks=callback_list)  #
            print('train,done!')
            """

            # 逐个batch训练,再更新
            for i in range(len(text_tag_CF_train_instances_tuple[0]) // Para.big_batch_size):
                print('train batch {}...'.format(i))
                batch_tuples, batch_labels = [a_array[i * Para.big_batch_size:(i + 1) * Para.big_batch_size] for a_array in
                                              text_tag_CF_train_instances_tuple], Para.train_labels[i * Para.big_batch_size:(i + 1) * Para.big_batch_size]
                hist = text_tag_CF_model.fit(batch_tuples, batch_labels, batch_size=Para.big_batch_size, epochs=1, verbose=1,
                                             shuffle=True)

                """
                # K.function()仍内存溢出
                text_features, tag_features = get_features(update_feature_instances)
                print('K func,done!')
                """

                if para_mode[-1] == '3':
                    # 如果是更新全部参数,就需要更新文本部分参数,batch更新
                    update_text_tag_CF_recommend_model_features(text_tag_CF_recommend_model, text_tag_CF_model,
                                                                update_feature_instances)
            # predict and evalute.
            csv_table_name = str(para_mode) + text_tag_CF_recommend_model.get_name()  if epoch == 0 else ""  # model.name

            evaluate_result = test_evalute(text_tag_CF_recommend_model, text_tag_CF_model, csv_table_name, 1,
                                           train=False)

            text_tag_CF_model.save_weights(
                Para.history_result_path + para_path_part.format(para_mode, epoch))  # 记录该epoch下的模型参数***

            if evaluate_result[0][3] > best_NDCG_5:  # 记录NDCG_5效果最好的模型下  提取的文本特征
                best_NDCG_5 = evaluate_result[0][3]
                best_epoch = epoch

        with open(Para.history_result_path + '/best_epoch_{}'.format(para_mode), 'w') as f:
            f.write(str(best_epoch))
        print('best epoch:{}'.format(best_epoch))

        # 返回效果最好的text_tag_CF_model
        best_model_path = Para.history_result_path + para_path_part.format(para_mode, best_epoch)
        text_tag_CF_model.load_weights(best_model_path)

        text_features, tag_features, text_tag_CF_model = update_text_tag_CF_recommend_model_features(
            text_tag_CF_recommend_model, text_tag_CF_model,
            update_feature_instances)

        np.savetxt(Para.text_features_path.format(para_mode), text_features)
        np.savetxt(Para.tag_features_path.format(para_mode), tag_features)

        return (text_features, tag_features), text_tag_CF_model


# 根据此时的text_tag_CF的模型参数，计算每个mashup的features，用于更新text_tag_CF_recommend_model的features参数
def update_text_tag_CF_recommend_model_features(text_tag_CF_recommend_model, text_tag_CF_model,
                                                update_feature_instances):
    update_feature_text_tag_model = Model(inputs=text_tag_CF_model.inputs,
                                          outputs=[text_tag_CF_model.get_layer('concatenate_1').input[0],
                                                   text_tag_CF_model.get_layer('concatenate_1').input[2]])
    text_features, tag_features = update_feature_text_tag_model.predict(update_feature_instances, verbose=0)
    print('predict new features,done!')

    # 是否可以通过改变self.feature的值改变model的train过程?
    features = np.hstack((text_features, tag_features))
    text_tag_CF_recommend_model.update_features(features)
    print('update features matrix,done!')
    return text_features, tag_features, text_tag_CF_model