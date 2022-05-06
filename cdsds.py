from surprise import Dataset
from surprise.model_selection import train_test_split
from time import sleep
import numpy as np
import random
import pandas as pd
from surprise import Reader
from collections import OrderedDict
from collections import defaultdict
from surprise import accuracy
import networkx as nx
import klcore
from tqdm import tqdm

def create_filmtrust_dataset(seed=19):
    random.seed(seed)
    print('dataset is filmtrust')
    percentage = 1  # preserve 'percentage' ratings.
    upItem = 5
    data = pd.read_pickle('Filmtrust.pkl')
    allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                  data[['userID', 'musicID', 'rating', 'ex']].values]
    userdct = {}
    for x in data[['userID', 'musicID', 'rating', 'ex']].values:
        if x[0] not in userdct.keys():
            userdct[x[0]] = 1
        else:
            userdct[x[0]] += 1
    SocialNet = nx.DiGraph()
    # data = pd.read_pickle('Filmtrust.pkl')
    # allusers = [int(x[0]) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
    # for userID in allusers:
    #     SocialNet.add_node(userID)
    edges = np.load('filmtrustEdges.npy', allow_pickle=True).tolist()
    SocialNet.add_edges_from(edges)
    testusers = []
    for user in SocialNet.nodes:
        if user in userdct.keys():
            if userdct[user] <= 5:  # cold user
                testusers.append(user)
    print(len(testusers))
    # testusers = sorted(list(SocialNet.nodes), key=lambda x: (SocialNet.in_degree(x) + SocialNet.out_degree(x)), reverse=False)[:256]
    allratings = sorted(allratings, key=lambda x: (int(x[0]), int(x[3])))
    allratingsDict = {}
    for element in allratings:
        user = element[0]
        item = element[1]
        rating = element[2]
        time = element[3]
        if user not in allratingsDict.keys():
            allratingsDict[user] = []
        allratingsDict[user].append((item, rating, time))
    allratingsDict = OrderedDict(sorted(allratingsDict.items()))
    #  FIXME: split 20% test users from whom items will be removed 80%.
    #  FIXME: Warning...
    #  FIXME: Warning...
    trainDict = {'userID': [], 'itemID': [], 'rating': []}
    testDict = {'userID': [], 'itemID': [], 'rating': []}
    for user in allratingsDict.keys():
        if user in testusers:  # test users
            for entry in allratingsDict[user]:
                if testDict['userID'].count(user) < 1:
                    testDict['userID'].append(user)
                    testDict['itemID'].append(entry[0])
                    testDict['rating'].append(entry[1])
                else:
                    if random.uniform(0, 1) <= percentage:
                        trainDict['userID'].append(user)
                        trainDict['itemID'].append(entry[0])
                        trainDict['rating'].append(entry[1])
        else:
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage:
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
    trainset = Dataset.load_from_df(pd.DataFrame(trainDict)[['userID', 'itemID', 'rating']],
                                    Reader(rating_scale=(1, 5))).build_full_trainset()
    testset = Dataset.load_from_df(pd.DataFrame(testDict)[['userID', 'itemID', 'rating']],
                                   Reader(rating_scale=(1, 5))).build_full_trainset().build_testset()
    return trainset, testset

def createdataset(seed=19, name='ml-100k'):
    print('dataset is ' + name)
    random.seed(seed)
    percentage = 0.2  # preserve 'percentage' ratings.
    testpercentage = 0.2
    upItem = 5
    if name == 'yahoo':
        percentage = 0.2
        data = pd.read_pickle('yahoo.pkl')
        allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
    elif name == 'filmtrust':
        # percentage = 1
        # data = pd.read_pickle('Filmtrust.pkl')
        # allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
        #                          data[['userID', 'musicID', 'rating', 'ex']].values]
        print('Shortcut!')
        return create_filmtrust_dataset(seed=seed)
    elif name == 'Epinions':
        percentage = 0.2
        data = pd.read_pickle('Epinions.pkl')
        allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                                 data[['userID', 'musicID', 'rating', 'ex']].values]
    else:
        data = Dataset.load_builtin(name)
        allratings = data.raw_ratings
    allratings = sorted(allratings, key=lambda x: (int(x[0]), int(x[3])))
    allratingsDict = {}
    for element in allratings:
        user = element[0]
        item = element[1]
        rating = element[2]
        time = element[3]
        if user not in allratingsDict.keys():
            allratingsDict[user] = []
        allratingsDict[user].append((item, rating, time))
    allratingsDict = OrderedDict(sorted(allratingsDict.items()))
    #  FIXME: split 20% test users from whom items will be removed 80%.
    #  FIXME: Warning...
    #  FIXME: Warning...
    trainDict = {'userID': [], 'itemID': [], 'rating': []}
    testDict = {'userID': [], 'itemID': [], 'rating': []}
    pbar = tqdm(total=len(allratingsDict.keys()))
    for user in allratingsDict.keys():
        if random.uniform(0, 1) <= testpercentage:  # test users
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage and trainDict['userID'].count(user) <= upItem:  # training data
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
                else:
                    if testDict['userID'].count(user) < 20:
                        if name not in ['Epinions']:
                            testDict['userID'].append(user)
                            testDict['itemID'].append(entry[0])
                            testDict['rating'].append(entry[1])
                        else:
                            if random.uniform(0, 1) <= 0.1:
                                testDict['userID'].append(user)
                                testDict['itemID'].append(entry[0])
                                testDict['rating'].append(entry[1])
        else:
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage:
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
        pbar.update(1)
    pbar.close()
    trainset = Dataset.load_from_df(pd.DataFrame(trainDict)[['userID', 'itemID', 'rating']],
                                    Reader(rating_scale=(1, 5))).build_full_trainset()
    testset = Dataset.load_from_df(pd.DataFrame(testDict)[['userID', 'itemID', 'rating']],
                                   Reader(rating_scale=(1, 5))).build_full_trainset().build_testset()
    return trainset, testset


# def precision_recall_at_k(predictions, k=10, threshold=3.5):
#     '''Return precision and recall at k metrics for each user.'''
#
#     # First map the predictions to each user.
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))
#
#     precisions = dict()
#     recalls = dict()
#     for uid, user_ratings in user_est_true.items():
#
#         # Sort user ratings by estimated value
#         user_ratings.sort(key=lambda x: x[0], reverse=True)
#
#         # Number of relevant items
#         n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
#
#         # Number of recommended items in top k
#         n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
#
#         # Number of relevant and recommended items in top k
#         n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
#                               for (est, true_r) in user_ratings[:k])
#
#         # Precision@K: Proportion of recommended items that are relevant
#         precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
#
#         # Recall@K: Proportion of relevant items that are recommended
#         recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
#
#     return precisions, recalls

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    pbar = tqdm(total=len(user_est_true.items()))
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = len(user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        pbar.update(1)
    pbar.close()
    return precisions, recalls


# def precision_recall_at_k(predictions, k=10, threshold=3.5):
#     '''Return precision and recall at k metrics for each user.'''
#
#     # First map the predictions to each user.
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))
#     precisions = dict()
#     recalls = dict()
#     n_rel = n_rec_k = n_rel_and_rec_k = 0
#     for uid, user_ratings in user_est_true.items():
#
#         # Sort user ratings by estimated value
#         user_ratings.sort(key=lambda x: x[0], reverse=True)
#
#         # Number of relevant items
#         n_rel += sum((true_r >= threshold) for (_, true_r) in user_ratings)
#
#         # Number of recommended items in top k
#         n_rec_k += len(user_ratings[:k])
#
#         # Number of relevant and recommended items in top k
#         n_rel_and_rec_k += sum((true_r >= threshold)
#                               for (_, true_r) in user_ratings[:k])
#
#     # Precision@K: Proportion of recommended items that are relevant
#     precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
#
#     # Recall@K: Proportion of relevant items that are recommended
#     recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
#
#     precisions['admin'] = precision
#     recalls['admin'] = recall
#
#     return precisions, recalls


def nppCalc(predictions):
    npp = 0
    for uid, _, true_r, est, _ in predictions:
        if np.fabs(est - true_r) <= 0.05:
            npp += 1
    return npp/len(predictions)

def create_testusers(trainset, testset):
    testusers = []
    for rawid, _, _ in testset:
        try:
            testusers.append(trainset.to_inner_uid(rawid))
        except:
            pass
        continue
    testusers = list(set(testusers))
    return testusers


class CalMetric(object):
    def __init__(self):
        self.maeList = []
        self.rmseList = []
        self.preList = []
        self.recList = []
        self.f1List = []
        self.totalmae = 0
        self.totalrmse = 0
        self.totalpre = 0
        self.totalrec = 0
        self.totalf1 = 0
        self.totalnpp = 0
        self.totalnsp = 0
        self.metricTensor = {}
        self.nppnsp = {}
        self.sim_options = {'name': 'cosine',
                       'user_based': True,  # compute similarities between users
                       'min_support': 0
                       }
    def cvcalculate(self, model, fold=5, neighbour=60):
        print('Using cross-validation')
        print('neighbour = ' + str(neighbour))
        for i in range(fold):
            print('Here is fold '+str(i+1))
            trainset, testset = createdataset(seed=19+i, name='ml-100k')
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            algo = model(neighbour, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.fit(trainset)
            predictions = algo.test(testset)
            predictions2 = algo.test2(testset)
            pnsp = algo.nspCalc()
            pnpp = nppCalc(predictions2)
            self.totalnpp += pnpp
            self.totalnsp += pnsp
            precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
            p = sum(prec for prec in precisions.values()) / len(precisions)
            r = sum(rec for rec in recalls.values()) / len(recalls)
            # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
            # p = precision
            # r = recall
            f = 2 * p * r / (p + r)
            self.totalmae += accuracy.mae(predictions)
            self.totalrmse += accuracy.rmse(predictions)
            self.totalpre += p
            self.totalrec += r
            self.totalf1 += f
        return self.totalmae/fold, self.totalrmse/fold, self.totalpre/fold, self.totalrec/fold, self.totalf1/fold, self.totalnpp/fold, self.totalnsp/fold


    def nocvcalculate(self, model, neighbour=60):
        print('No cross-validation')
        trainset, testset = createdataset(name='Epinions')
        testusers = []
        for rawid, _, _ in testset:
            try:
                testusers.append(trainset.to_inner_uid(rawid))
            except:
                pass
            continue
        testusers = list(set(testusers))
        algo = model(neighbour, 1, sim_options=self.sim_options, verbose=True)
        algo.testusers = testusers
        algo.trust_neighbours, Communities = klcore.get_trust_neighbours(trainset=trainset, testset=testset)
        algo.fit(trainset)
        print('Fitting finished')
        algo.sim = klcore.refine4(algo.sim, Communities, trainset, testset)
        predictions = algo.test(testset)
        print('Testing finished')
        # predictions2 = algo.test2(testset)
        # pnsp = algo.nspCalc()
        # pnpp = nppCalc(predictions2)
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
        p = sum(prec for prec in precisions.values()) / len(precisions)
        r = sum(rec for rec in recalls.values()) / len(recalls)
        # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
        # p = precision
        # r = recall
        f = 2*p*r/(p+r)
        return accuracy.mae(predictions), accuracy.rmse(predictions), p, r, f


    def Curvecvcalculate(self, model, fold=5, neighbours=None):
        print('Using Curve cross-validation')
        self.metricTensor = {'mae': np.zeros((fold, len(neighbours))), 'rmse': np.zeros((fold, len(neighbours))), 'pre': np.zeros((fold, len(neighbours))), 'rec': np.zeros((fold, len(neighbours))), 'f1': np.zeros((fold, len(neighbours)))}
        self.nppnsp = {'npp': np.zeros((fold, len(neighbours))), 'nsp': np.zeros((fold, len(neighbours)))}
        for i in range(fold):
            print('Here is fold '+str(i+1))
            trainset, testset = createdataset(seed=19+i, name='yahoo')
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            algo = model(60, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.fit(trainset)
            for j, neighbour in enumerate(neighbours):
                print('Here neighbour is '+str(neighbour))
                print('Here is ' + str(j+1) + '/' + str(len(neighbours)))
                algo.k = neighbour
                print(algo.k)
                predictions = algo.test(testset)
                predictions2 = algo.test2(testset)
                pnsp = algo.nspCalc()
                pnpp = nppCalc(predictions2)
                precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                p = sum(prec for prec in precisions.values()) / len(precisions)
                r = sum(rec for rec in recalls.values()) / len(recalls)
                # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
                # p = precision
                # r = recall
                f = 2 * p * r / (p + r)
                self.nppnsp['npp'][i][j] = pnpp
                self.nppnsp['nsp'][i][j] = pnsp
                self.metricTensor['mae'][i][j] = accuracy.mae(predictions)
                self.metricTensor['rmse'][i][j] = accuracy.rmse(predictions)
                self.metricTensor['pre'][i][j] = p
                self.metricTensor['rec'][i][j] = r
                self.metricTensor['f1'][i][j] = f
        return self.metricTensor['mae'].mean(axis=0), self.metricTensor['rmse'].mean(axis=0), self.metricTensor['pre'].mean(axis=0), self.metricTensor['rec'].mean(axis=0), self.metricTensor['f1'].mean(axis=0), self.nppnsp['npp'].mean(axis=0), self.nppnsp['nsp'].mean(axis=0)

    def bigcCurvecvcalculate(self, model, fold=5, clist=None, name='ml-100k'):  # only for diri model
        print('Using bigC cross-validation')
        self.metricTensor = {'mae': np.zeros((fold, len(clist))), 'rmse': np.zeros((fold, len(clist))),
                             'pre': np.zeros((fold, len(clist))), 'rec': np.zeros((fold, len(clist))),
                             'f1': np.zeros((fold, len(clist)))}
        self.nppnsp = {'npp': np.zeros((fold, len(clist))), 'nsp': np.zeros((fold, len(clist)))}
        for i in range(fold):
            print('Here is fold ' + str(i + 1))
            trainset, testset = createdataset(seed=19 + i, name=name)
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            algo = model(60, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            for j, bigc in enumerate(clist):
                print('Here cVal is ' + str(bigc))
                print('Here is ' + str(j + 1))
                algo.bigC = bigc
                print(algo.bigC)
                algo.fit(trainset)
                predictions = algo.test(testset)
                predictions2 = algo.test2(testset)
                pnsp = algo.nspCalc()
                pnpp = nppCalc(predictions2)
                precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                p = sum(prec for prec in precisions.values()) / len(precisions)
                r = sum(rec for rec in recalls.values()) / len(recalls)
                # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
                # p = precision
                # r = recall
                f = 2 * p * r / (p + r)
                self.nppnsp['npp'][i][j] = pnpp
                self.nppnsp['nsp'][i][j] = pnsp
                self.metricTensor['mae'][i][j] = accuracy.mae(predictions)
                self.metricTensor['rmse'][i][j] = accuracy.rmse(predictions)
                self.metricTensor['pre'][i][j] = p
                self.metricTensor['rec'][i][j] = r
                self.metricTensor['f1'][i][j] = f
        return self.metricTensor['mae'].mean(axis=0), self.metricTensor['rmse'].mean(axis=0), self.metricTensor[
            'pre'].mean(axis=0), self.metricTensor['rec'].mean(axis=0), self.metricTensor['f1'].mean(axis=0), \
               self.nppnsp['npp'].mean(axis=0), self.nppnsp['nsp'].mean(axis=0)

    def bigCCurvenocvcalculate(self, model, neighbour=60, clist=None, name='ml-100k'):
        trainset, testset = createdataset(name=name)
        testusers = []
        for rawid, _, _ in testset:
            try:
                testusers.append(trainset.to_inner_uid(rawid))
            except:
                pass
            continue
        testusers = list(set(testusers))
        for bigC in clist:
            algo = model(neighbour, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.bigC = bigC
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
            p = sum(prec for prec in precisions.values()) / len(precisions)
            r = sum(rec for rec in recalls.values()) / len(recalls)
            # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
            # p = precision
            # r = recall
            f = 2*p*r/(p+r)
            self.maeList.append(accuracy.mae(predictions))
            self.rmseList.append(accuracy.rmse(predictions))
            self.preList.append(p)
            self.recList.append(r)
            self.f1List.append(f)
        return self.maeList, self.rmseList, self.preList, self.recList, self.f1List


    def bigCCurvenbnocvcalculate(self, model, neighbour=None, clist=None, name='ml-100k'):
        trainset, testset = createdataset(name=name)
        tempDict = defaultdict(dict)
        testusers = []
        for rawid, _, _ in testset:
            try:
                testusers.append(trainset.to_inner_uid(rawid))
            except:
                pass
            continue
        testusers = list(set(testusers))
        for bigC in clist:
            algo = model(60, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.bigC = bigC
            algo.fit(trainset)
            for nb in neighbour:
                algo.k = nb
                predictions = algo.test(testset)
                precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                p = sum(prec for prec in precisions.values()) / len(precisions)
                r = sum(rec for rec in recalls.values()) / len(recalls)
                f = 2*p*r/(p+r)
                tempDict[nb][bigC] = {}
                tempDict[nb][bigC]['mae'] = accuracy.mae(predictions)
                tempDict[nb][bigC]['rmse'] = accuracy.rmse(predictions)
                tempDict[nb][bigC]['pre'] = p
                tempDict[nb][bigC]['rec'] = r
                tempDict[nb][bigC]['f1'] = f
        return tempDict


    def bigcCurvenbcvcalculate(self, model, neighbour=None, fold=5, clist=None, name='ml-100k'):  # only for diri model
        tempDict = defaultdict(dict)
        for cc in clist:
            for nbnb in neighbour:
                tempDict[nbnb][cc] = {}
                tempDict[nbnb][cc]['mae'] = 0
                tempDict[nbnb][cc]['rmse'] = 0
                tempDict[nbnb][cc]['pre'] = 0
                tempDict[nbnb][cc]['rec'] = 0
                tempDict[nbnb][cc]['f1'] = 0
        for i in range(fold):
            print('Here is fold ' + str(i + 1))
            trainset, testset = createdataset(seed=19 + i, name=name)
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            for bigC in clist:
                print('Here is fold ' + str(i + 1) + ', bigC is ' + str(bigC))
                algo = model(60, 1, sim_options=self.sim_options, verbose=True)
                algo.testusers = testusers
                algo.bigC = bigC
                algo.fit(trainset)
                for nb in neighbour:
                    algo.k = nb
                    predictions = algo.test(testset)
                    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                    p = sum(prec for prec in precisions.values()) / len(precisions)
                    r = sum(rec for rec in recalls.values()) / len(recalls)
                    f = 2 * p * r / (p + r)
                    tempDict[nb][bigC]['mae'] += accuracy.mae(predictions)
                    tempDict[nb][bigC]['rmse'] += accuracy.rmse(predictions)
                    tempDict[nb][bigC]['pre'] += p
                    tempDict[nb][bigC]['rec'] += r
                    tempDict[nb][bigC]['f1'] += f
        for cc in clist:
            for nbnb in neighbour:
                tempDict[nbnb][cc]['mae'] /= 5
                tempDict[nbnb][cc]['rmse'] /= 5
                tempDict[nbnb][cc]['pre'] /= 5
                tempDict[nbnb][cc]['rec'] /= 5
                tempDict[nbnb][cc]['f1'] /= 5
        return tempDict


    def oldcvcalculate(self, model, fold=5, neighbour=60):
        print('Using cross-validation')
        print('neighbour = ' + str(neighbour))
        for i in range(fold):
            print('Here is fold '+str(i+1))
            trainset, testset = createdataset(seed=19+i, name='ml-100k')
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            algo = model(neighbour, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
            p = sum(prec for prec in precisions.values()) / len(precisions)
            r = sum(rec for rec in recalls.values()) / len(recalls)
            # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
            # p = precision
            # r = recall
            f = 2 * p * r / (p + r)
            self.totalmae += accuracy.mae(predictions)
            self.totalrmse += accuracy.rmse(predictions)
            self.totalpre += p
            self.totalrec += r
            self.totalf1 += f
        return self.totalmae/fold, self.totalrmse/fold, self.totalpre/fold, self.totalrec/fold, self.totalf1/fold

    def ComCurvecvcalculate(self, model, fold=5, neighbours=None):
        print('Using Curve cross-validation')
        self.metricTensor = {'mae': np.zeros((fold, len(neighbours))), 'rmse': np.zeros((fold, len(neighbours))), 'pre': np.zeros((fold, len(neighbours))), 'rec': np.zeros((fold, len(neighbours))), 'f1': np.zeros((fold, len(neighbours)))}
        self.nppnsp = {'npp': np.zeros((fold, len(neighbours))), 'nsp': np.zeros((fold, len(neighbours)))}
        for i in range(fold):
            print('Here is fold '+str(i+1))
            trainset, testset = createdataset(seed=19+i, name='Epinions')
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            algo = model(60, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.trust_neighbours, Communities = klcore.get_trust_neighbours(trainset=trainset, testset=testset)
            algo.current_fold = 19+i
            algo.fit(trainset)
            # algo.sim = klcore.refine1(algo.sim, Communities, trainset, testset)
            for j, neighbour in enumerate(neighbours):
                print('Here neighbour is '+str(neighbour))
                print('Here is ' + str(j+1) + '/' + str(len(neighbours)))
                algo.k = neighbour
                print(algo.k)
                predictions = algo.test(testset)
                predictions2 = algo.test2(testset)
                pnsp = algo.nspCalc()
                pnpp = nppCalc(predictions2)
                precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                p = sum(prec for prec in precisions.values()) / len(precisions)
                r = sum(rec for rec in recalls.values()) / len(recalls)
                # precision, recall = precision_recall_at_k(predictions, k=10, threshold=4)
                # p = precision
                # r = recall
                f = 2 * p * r / (p + r)
                self.nppnsp['npp'][i][j] = pnpp
                self.nppnsp['nsp'][i][j] = pnsp
                self.metricTensor['mae'][i][j] = accuracy.mae(predictions)
                self.metricTensor['rmse'][i][j] = accuracy.rmse(predictions)
                self.metricTensor['pre'][i][j] = p
                self.metricTensor['rec'][i][j] = r
                self.metricTensor['f1'][i][j] = f
        return self.metricTensor['mae'].mean(axis=0), self.metricTensor['rmse'].mean(axis=0), self.metricTensor['pre'].mean(axis=0), self.metricTensor['rec'].mean(axis=0), self.metricTensor['f1'].mean(axis=0), self.nppnsp['npp'].mean(axis=0), self.nppnsp['nsp'].mean(axis=0)


    def clearmetric(self):
        self.totalmae = 0
        self.totalrmse = 0
        self.totalpre = 0
        self.totalrec = 0
        self.totalf1 = 0
        self.sim_options = {'name': 'cosine',
                            'user_based': True,  # compute similarities between users
                            'min_support': 0
                            }


# if __name__ == '__main__':
#     create_filmtrust_dataset()