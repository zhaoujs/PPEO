# encoding: utf-8
import copy

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score as G_mean

import archiving_neg
import init_pos
import init_neg
import update_neg
import update_pos
from SSHR_utils import Kmean_Train
from fitness_funs import *
from sec import *
from tool import *
from SSHR_utils import *



def SSHC(train_x, train_y, test_x, test_y, min_label):
    minority_index_list = list(np.where(train_y == min_label)[0])
    cluster_list = []

    train_k = Kmean_Train(train_x, train_y, maxnum=20, weight=0.3)
    test_k = Kmean_Test(test_x, test_y)
    k_center(train_k)
    kmean_train_new_1120(train_k, test_k)
    iterative_update2_1(train_k, test_k, 0)
    kmean_predict0(train_k)
    kmean_predict_testdata0(train_k, test_k)

    for i in range(train_k.k):
        cluster_list.append([])
    train_cluster = train_k.cluster
    for minority_index_ in minority_index_list:
        cluster_list[train_cluster[minority_index_]].append(minority_index_)

    res_cluster_list = []
    res_y_c = []
    res_cluster_center = []
    for index, item in enumerate(cluster_list):
        print(len(item), end="\t")
        if len(item) != 0:
            res_cluster_list.append(cluster_list[index])
            res_y_c.append(train_k.y_c[index])
            res_cluster_center.append(train_k.x_c[index])
    return res_cluster_list, res_y_c, res_cluster_center, train_k



class DoubleMopso:
    def __init__(self, particals, w, c1, c2, thresh, x, y, min_label, mesh_div=10, model=SVC()):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div  # 网格等分数量
        self.particals = particals  # 粒子群的数量
        self.thresh = thresh  # 外部存档阀值
        self.min_label = min_label
        # 初始两个 mopso 种群
        self.model = model
        x, x_val, y, y_val = x, x, y, y
        self.x = x  # 传入的x
        self.y = y  # 传入的y
        self.x_val = x_val  #
        self.y_val = y_val  #

    def Neg_PSO(self):
        min_ = np.array([0] * len(self.y[self.y != self.min_label]))  # 粒子坐标的最小值
        max_ = np.array([1] * len(self.y[self.y != self.min_label]))  # 粒子坐标的最大值
        self.PSO_neg = NegMopso(self.particals, self.w, self.c1, self.c2, max_, min_, self.thresh, self.x, self.y,
                                self.x_val, self.y_val,
                                self.min_label, self.mesh_div, self.model)

    def Pos_PSO(self):

        self.PSO_pos = PosMopso(self.particals, self.w, self.c1, self.c2, 0, 0, self.thresh, self.x, self.y,
                                self.x_val, self.y_val,
                                self.min_label, self.mesh_div, self.model)

    def initialize(self):
        self.Neg_data = self.x[self.y != self.min_label]
        self.Pos_data = self.x[self.y == self.min_label]
        self.PSO_pos.initialize()


    def update_(self):
        self.PSO_pos.update_()



    def SSHC_Regin_Best_(self, Query_x):
        centroids = self.PSO_pos.cluster_center
        distances = np.linalg.norm(centroids - Query_x, axis=1)
        closest_centroid_index = np.argmin(distances)
        centroids_all = self.PSO_pos.SSHC_K.x_c
        distances_all = np.linalg.norm(centroids_all - Query_x, axis=1)
        closest_centroid_index_all = np.argmin(distances_all)
        if distances[closest_centroid_index] != distances_all[closest_centroid_index_all]:
            return -1
        else:
            return closest_centroid_index
    def done_v3(self, cycle_):
        self.Pos_PSO()
        # 初始化
        self.initialize()

        for i1 in range(cycle_):
            self.update_()
            print(f"第{i1}轮\n "
                  f"Pos:{self.PSO_pos.archive_fitness}"
                  )
            # return self.archive_in, self.archive_fitness
            # 生成一个集成的model
            # 这边表示Neg_PSO是单目标的。
        res_list = []
        res_train_list = []
        res_model_list = []
        data_last = []
        y_last = []
        for i2 in range(len(self.PSO_pos.archive_syn)):
            Neg_data = self.Neg_data
            Pos_data = self.Pos_data
            Syn_data = self.PSO_pos.archive_syn[i2]
            model = self.model
            all_data = np.array(Neg_data).tolist() + np.array(Pos_data).tolist() + np.array(Syn_data).tolist()
            all_data = np.array(all_data)
            # all_data = np.concatenate((Neg_data, Pos_data, Syn_data), axis=0)
            all_data_y = [-1] * len(Neg_data) + [self.min_label] * len(Pos_data) + [self.min_label] * len(Syn_data)
            data_last.append(all_data)
            y_last.append(all_data_y)


        return data_last, y_last



class NegMopso:
    def __init__(self, particals, w, c1, c2, max_, min_, thresh, x, y, x_val, y_val, min_label, mesh_div=10,
                 model=SVC()):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.max_v = (max_ - min_) * 0.05  # 速度下限
        self.min_v = (max_ - min_) * 0.05 * (-1)  # 速度上限

        self.x = x  # 传入的x
        self.y = y  # 传入的y
        self.x_val = x_val  #
        self.y_val = y_val  #
        self.min_label = min_label  # 传入的
        self.neg_g = None
        self.syn_g = None
        self.Model = model

    def evaluation_fitness(self):
        fitness_curr = []
        bar = tqdm(range(self.in_.shape[0]))
        for i in bar:
            res = self.getfitness_(i)
            fitness_curr.append(res)
            bar.set_description(f"欠采样No.{i}")
        self.fitness_ = np.array(fitness_curr)  # 适应值

    def initialize(self):
        self.Neg_threshold = 0.5
        self.in_ = init_neg.init_designparams(self.particals, self.min_, self.max_)
        self.choose = copy.deepcopy(self.in_)
        self.choose[self.choose > self.Neg_threshold] = 1
        self.choose[self.choose <= self.Neg_threshold] = 0
        self.v_ = init_neg.init_v(self.particals, self.min_v, self.max_v)
        self.Neg_data = self.x[self.y != self.min_label]
        self.Pos_data = self.x[self.y == self.min_label]
        self.evaluation_fitness()
        self.in_p, self.fitness_p = init_neg.init_pbest(self.in_, self.fitness_)
        self.archive_in, self.archive_fitness = init_neg.init_archive(self.in_, self.fitness_)
        self.in_g, self.fitness_g = init_neg.init_gbest(self.archive_in, self.archive_fitness,
                                                        self.mesh_div, self.min_, self.max_, self.particals)
        self.neg_g = []
        Neg_g = copy.deepcopy(self.in_g)
        for i in Neg_g:
            G_best_data_item = self.Neg_data[i > self.Neg_threshold]
            self.neg_g.append(G_best_data_item)

    def update_(self):
        self.v_ = update_neg.update_v(self.v_, self.min_v, self.max_v, self.in_, self.in_p, self.in_g, self.w, self.c1,
                                      self.c2)
        self.in_ = update_neg.update_in(self.in_, self.v_, self.min_, self.max_)
        self.choose = copy.deepcopy(self.in_)
        self.choose[self.choose > self.Neg_threshold] = 1
        self.choose[self.choose <= self.Neg_threshold] = 0
        self.evaluation_fitness()
        self.in_p, self.fitness_p = update_neg.update_pbest(self.in_, self.fitness_, self.in_p, self.fitness_p)
        self.archive_in, self.archive_fitness = update_neg.update_archive(self.in_, self.fitness_, self.archive_in,
                                                                          self.archive_fitness, self.thresh,
                                                                          self.mesh_div,
                                                                          self.min_, self.max_, self.particals)
        self.in_g, self.fitness_g = update_neg.update_gbest(self.archive_in, self.archive_fitness, self.mesh_div,
                                                            self.min_,
                                                            self.max_, self.particals)
        self.neg_g = []
        Neg_g = copy.deepcopy(self.in_g)
        for i in Neg_g:
            G_best_data_item = self.Neg_data[i > self.Neg_threshold]
            self.neg_g.append(G_best_data_item)

    def done(self, cycle_):
        self.initialize()
        for i in range(cycle_):
            self.update_()
        return self.archive_in, self.archive_fitness
    def getfitness_(self, index):
        partical_item = self.choose[index]
        all_data = self.Neg_data[partical_item == 1]

        choose_instance = len(all_data)
        if choose_instance == 0:
            return [np.inf, np.inf]
        all_data = np.concatenate((all_data, self.Pos_data), axis=0)
        all_data_lable = np.array(choose_instance * [-1] + [self.min_label] * len(self.Pos_data))
        if self.syn_g is not None:
            all_data_lable = np.concatenate((all_data_lable, len(self.syn_g[index]) * [self.min_label]), axis=0)
            all_data = np.array(all_data).tolist() + np.array(self.syn_g[index]).tolist()
            all_data = np.array(all_data)
        Model = self.Model
        Model.fit(all_data, all_data_lable)
        pred = Model.predict(self.x_val)
        GM = G_mean(self.y_val, pred)
        return [1 - GM]


class PosMopso:
    def __init__(self, particals, w, c1, c2, max_, min_, thresh, x, y, x_val, y_val, min_label, mesh_div=10,
                 model=SVC()):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.x = x  # 传入的x
        self.y = y  # 传入的y
        self.x_val = x_val  #
        self.y_val = y_val  #
        self.min_label = min_label
        self.neg_g = None
        self.syn_g = None
        self.Model = model
        self.K = 5
        # 获得多数类的样本集合
        self.Neg_data = self.x[self.y != self.min_label]
        self.Pos_data = self.x[self.y == self.min_label]

        cluster_list1, cluster_label, X_C, SSHC_K = SSHC(x, y, x_val, y_val, min_label)

        self.cluster_list = cluster_list1
        self.cluster_label = cluster_label
        self.k = int(len(self.Pos_data) ** 0.5)
        self.cluster_center = X_C
        self.SSHC_K = SSHC_K
        min_ = np.array([0] * len(self.cluster_list))  # 粒子坐标的最小值
        max_ = np.array([1] * len(self.cluster_list))  # 粒子坐标的最大值
        self.max_ = max_
        self.min_ = min_
        self.max_v = (max_ - min_) * 0.05  # 速度下限
        self.min_v = (max_ - min_) * 0.05 * (-1)  # 速度上限

    def evaluation_fitness(self):

        fitness_curr = []

        syn_curr = []
        bar = tqdm(range(self.in_.shape[0]))
        for i in bar:
            res, syn = self.getfitness_(i)
            fitness_curr.append(res)
            syn_curr.append(syn)
            bar.set_description(f"过采样No.{i}")
        self.fitness_ = np.array(fitness_curr)
        self.syn_ = syn_curr

    def initialize(self):

        self.in_ = init_pos.init_designparams(self.particals, self.min_, self.max_)

        self.v_ = init_pos.init_v(self.particals, self.min_v, self.max_v)


        self.evaluation_fitness()

        self.in_p, self.fitness_p, self.syn_p = init_pos.init_pbest(self.in_, self.fitness_, self.syn_)

        self.archive_in, self.archive_fitness, self.archive_syn = init_pos.init_archive(self.in_, self.fitness_,
                                                                                        self.syn_)

        self.in_g, self.fitness_g, self.syn_g = init_pos.init_gbest(self.archive_in, self.archive_fitness,
                                                                    self.archive_syn,
                                                                    self.mesh_div, self.min_, self.max_, self.particals)

    def update_(self):
        self.v_ = update_pos.update_v(self.v_, self.min_v, self.max_v, self.in_, self.in_p, self.in_g, self.w, self.c1,
                                      self.c2)
        self.in_ = update_pos.update_in(self.in_, self.v_, self.min_, self.max_)
        self.evaluation_fitness()
        # 更新pbest
        self.in_p, self.fitness_p, self.syn_p = update_pos.update_pbest(self.in_, self.fitness_,
                                                                        self.syn_, self.in_p, self.fitness_p,
                                                                        self.syn_p)
        self.archive_in, self.archive_fitness, self.archive_syn \
            = update_pos.update_archive(self.in_, self.fitness_, self.syn_,
                                        self.archive_in, self.archive_fitness, self.archive_syn,
                                        self.thresh, self.mesh_div,
                                        self.min_, self.max_, self.particals)
        # 更新gbest
        self.in_g, self.fitness_g, self.syn_g = (
            update_pos.update_gbest(self.archive_in, self.archive_fitness, self.archive_syn,
                                    self.mesh_div, self.min_, self.max_, self.particals))

    def done(self, cycle_):
        self.initialize()
        for i in range(cycle_):
            self.update_()
        return self.archive_in, self.archive_fitness

    def getfitness_(self, p_index):
        Model = self.Model
        Neg_data = self.Neg_data
        Pos_weight = self.in_[p_index]
        self.Pos_data = []
        self.Response_Pos = {}
        i = 0
        for cluster_item in self.cluster_list:
            for cluster_item_item in cluster_item:
                self.Pos_data.append(self.x[cluster_item_item])
                self.Response_Pos[i] = cluster_item_item
                i += 1
        all_data = np.concatenate((self.Pos_data, Neg_data), axis=0)
        all_data_y = [self.min_label] * len(self.Pos_data) + [-1] * len(Neg_data)

        pro_x, pro_y, syn = self.SMOTE_SPA_SYN(all_data, all_data_y, Pos_weight)
        Model.fit(pro_x, pro_y)
        train_pred = Model.predict(self.x)
        fitness_ls = CalRegion(self.y, train_pred, self.cluster_list)
        res = []
        for i in fitness_ls[1:]:
            res_item = fitness_ls[0][0] * i[0]
            res.append(1 - res_item)
        return res, syn



    def SMOTE_SPA_SYN(self, x, y, weight):
        X_minority = x[: len(self.Pos_data)]
        X_majority = x[len(self.Pos_data):]
        K = self.K
        if len(X_minority) < K + 1:
            K = int(len(X_minority) ** 0.5)
        knn = KNN(k=K + 1)
        knn.fit(X_minority)
        knn2 = KNN(k=1)
        knn2.fit(X_majority)
        Pos_index = []
        for index, item in enumerate(X_minority):
            distance, samples, samples_index = knn.predict(item)
            distance, samples, samples_index = distance[1:], samples[1:], samples_index[1:]
            ls_dict = {}
            ls_dict["index"] = index
            ls_dict["sample"] = item
            ls_dict["distance"] = distance
            ls_dict["KNN_Sample"] = samples
            ls_dict["Sample_index"] = samples_index
            ls_dict["KNN_nei_label"] = np.array(y)[ls_dict["Sample_index"]]
            distance, samples, samples_index = knn2.predict(item)
            distance, samples, samples_index = distance[0], samples[0], samples_index[0]
            ls_dict["KNN2_SPA_distance"] = distance
            ls_dict["KNN2_SPA_samples"] = samples
            ls_dict["KNN2_SPA_samples_index"] = samples_index
            Pos_index.append(ls_dict)
        choose_weight = copy.deepcopy(weight)
        choose_weight = choose_weight / np.sum(choose_weight)
        num = Counter(y)[-1] - Counter(y)[self.min_label]
        syn = []
        cluster_list = self.cluster_list

        for i in range(num):
            if len(cluster_list) == 1:
                choose_Pos_regin_list = cluster_list[0]
            else:
                choose_Pos_regin_list_idx = np.random.choice(len(cluster_list), p=choose_weight)
                choose_Pos_regin_list = cluster_list[choose_Pos_regin_list_idx]
            chosen_index = cluster_list.index(choose_Pos_regin_list)
            choose_Pos_index0 = np.random.choice(choose_Pos_regin_list)
            choose_Pos_index = get_key(self.Response_Pos, choose_Pos_index0)
            choose_Pos = Pos_index[choose_Pos_index]
            src_item = choose_Pos["sample"]
            if self.cluster_label[chosen_index] == self.min_label:
                nei_samples_index = np.random.choice(len(choose_Pos["KNN_Sample"]))
                nei_samples = choose_Pos["KNN_Sample"][nei_samples_index]
                alpha = np.random.uniform(0, 1)
                syn_item = src_item + alpha * (nei_samples - src_item)
                syn.append(syn_item)
            else:
                r = choose_Pos["KNN2_SPA_distance"]
                new_item = generate_random_point_around_a(src_item, r)
                syn.append(new_item)
        X_all = np.concatenate((x, np.array(syn)))
        y_all = np.concatenate((y, np.array([1] * num)))
        return X_all, y_all, syn


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return None
