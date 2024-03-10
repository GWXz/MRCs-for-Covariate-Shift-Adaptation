import numpy as np
from sklearn.neighbors import NearestNeighbors
from CovShiftGen import CovShiftGen


def main():
    # 引入基于特征中位数的协变量移位
    # For Mac
    path = 'C:/Users/a/PycharmProjects/dw/MRCs-for-Covariate-Shift-Adaptation-main/'
    add_paths = [
        '/Users/<username>/cvx'
        'Datasets/',
        'Auxiliary_Functions/',
        'DWGCS/',
    ]

    # Add paths
    import sys
    sys.path.append(path)
    for add_path in add_paths:
        sys.path.append(add_path)

    from DWGCS import DWGCS

    dataset = np.genfromtxt('/home/zhengxi/code/MRCs-for-Covariate-Shift-Adaptation-main/Datasets/Blood.csv',
                            delimiter=',')
    dataset_normalize = np.genfromtxt('/home/zhengxi/code/MRCs-for-Covariate-Shift-Adaptation-main'
                                      '/Datasets/Blood_normalize.csv', delimiter=',')
    d = dataset_normalize.shape[1]
    X = dataset_normalize[:, :d - 1]

    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    sigma_ = np.mean(distances[:, 49])

    class BaseMdl:
        def __init__(self, intercep, deterministic, feature_map, labels, lambda0, loss, sigma_):
            self.intercep = intercep
            self.deterministic = deterministic
            self.feature_map = feature_map
            self.labels = labels
            self.lambda0 = lambda0
            self.loss = loss
            self.sigma_ = sigma_

    feature = 1
    # [Train_Set,Test_Set,n,t] = CovShiftGen.Features_BreastCancer(dataset, dataset_normalize, feature)
    [Train_Set, Test_Set, n, t] = CovShiftGen.Features(dataset_normalize, feature)  # 协变量移位

    xtr = Train_Set[:, :d - 1]
    ytr = Train_Set[:, d - 1:].astype(int)
    xte = Test_Set[:, :d - 1]
    yte = Test_Set[:, d - 1:].astype(int)

    # DWGCS 0-1-loss
    D = 1.0 / np.square(1.0 - (np.arange(0.0, 1.0, 0.1)))
    RU_Dwgcs = np.zeros(len(D))

    Dwgcs = []
    for l in range(len(D)):
        MdlAux = BaseMdl(False, True, 'linear', 2, 0, '0-1', sigma_)
        MdlAux.D = D[l]
        Dwgcs.append(MdlAux)
        Dwgcs[l] = DWGCS.DWGCS.DWKMM(Dwgcs[l], xtr, xte)  # 利用双重加权
        Dwgcs[l] = DWGCS.DWGCS.parameters(Dwgcs[l], xtr, ytr, xte)  # 得到向量
        Dwgcs[l] = DWGCS.DWGCS.learning(Dwgcs[l], xte)  # 获取分类器参数
        RU_Dwgcs[l] = Dwgcs[l].RU

    RU_best_Dwgcs = np.min(RU_Dwgcs)
    position = np.argmin(RU_Dwgcs)
    Dwgcs[position] = DWGCS.DWGCS.prediction(Dwgcs[position], xte, yte)  # 预测并给出误差
    error_best_Dwgcs = Dwgcs[position].error


if __name__ == '__main__':
    main()
