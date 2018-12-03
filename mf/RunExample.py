import os

import matplotlib.pyplot as plt
import numpy as np
from mf.LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from mf.ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    file_path = "../data/test_pmf.data"
    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 2, "num_batches": 100,
                    "batch_size": 1000})
    ratings = load_rating_data(file_path)
    print((len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat))
    train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.fit(train, test)
    pmf.save_UI('../data')

    # Check performance by plotting train and test errors
    plt.plot(list(range(pmf.maxepoch)), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(list(range(pmf.maxepoch)), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print(("precision_acc,recall_acc:" + str(pmf.topK(test))))


def get_UI_factors(inner_train_mashup_api_list,num_feat=25):
    root_path = os.path.abspath('..')
    pmf = PMF()
    pmf.set_params({"num_feat":num_feat, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
    w_User, w_Item =None,None
    if not (os.path.exists(root_path + '/data/U.factors')):
        train, test = train_test_split(inner_train_mashup_api_list, test_size=0.2)  # spilt_rating_dat(ratings)
        w_User, w_Item = pmf.fit(train, test)
        pmf.save_UI(root_path + '/data')
    else:
        w_User, w_Item = pmf.read_UI(root_path + '/data')
    print('get U,V factors of PMF, done!')
    return w_User, w_Item