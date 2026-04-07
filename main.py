

from mopso_regin import *

from sklearn.neural_network import MLPClassifier

def loadfile(filename):
    x, y = load_svmlight_file(filename)
    return np.array(x.todense()), np.array(y.astype(np.int32))
def Stratified_fold_K_version_2(x, y, n_spli=5, use_file=None, random_sta=42):
    skf = StratifiedKFold(n_splits=n_spli, shuffle=True, random_state=random_sta)
    i = 0
    if use_file != None:
        try:
            os.mkdir("{}".format(use_file))
        except OSError as error:
            pass
        for train_index, test_index in skf.split(x, y):
            i = i + 1
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            savelibsvm(X_train, y_train, "{}/{}train.txt".format(use_file, i))
            savelibsvm(X_test, y_test, "{}/{}test.txt".format(use_file, i))
    else:
        res = {}
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for train_index, test_index in skf.split(x, y):
            i = i + 1
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_x.append(copy.deepcopy(X_train))
            train_y.append(copy.deepcopy(y_train))
            test_x.append(copy.deepcopy(X_test))
            test_y.append(copy.deepcopy(y_test))
        res["train_x"] = train_x
        res["train_y"] = train_y
        res["test_x"] = test_x
        res["test_y"] = test_y
        return res


def run_item(data_name, No, train_x, train_y):

    w = 0.7298  # 惯性因子
    c1 = 1.49  # 局部速度因子
    c2 = 1.49  # 全局速度因子
    particals = 15  # 粒子群的数量
    cycle_ = 30  # 迭代次数
    mesh_div = 3  # 网格等分数量
    thresh = 20  # 外部存档阀值
    min_label = 1

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    mopso_ = DoubleMopso(particals, w, c1, c2, thresh, train_x, train_y, min_label, mesh_div, model=model)
    x_list, y_list = mopso_.done_v3(cycle_)



if __name__ == '__main__':

    filename = "ecoli4.txt"
    x, y = loadfile(filename)
    mm = StandardScaler()
    x = mm.fit_transform(x)
    all_data = Stratified_fold_K_version_2(x, y, n_spli=5, random_sta=42)
    No = 0
    train_x, train_y = all_data["train_x"][No], all_data["train_y"][No]
    run_item(filename, No, train_x, train_y)

