import numpy as np
from surprise import PredictionImpossible
from six import iteritems
from surprise import AlgoBase
import heapq
from statistics import median
from scipy.stats import entropy
from statistics import stdev
from surprise.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from scipy.stats import dirichlet
import collections
from tqdm import tqdm
import math
from surprise import accuracy
# from surprise.prediction_algorithms.predictions import Prediction
from collections import OrderedDict



class mySymmetricAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items   # |U|or|I|
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users   # |U|or|I|
        self.xr = self.trainset.ur if ub else self.trainset.ir   # user ratings or item ratings
        self.yr = self.trainset.ir if ub else self.trainset.ur   # user ratings or item ratings

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class diri(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.bigC = 3
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])
        print('Computing item similarity matrix...')
        itemSimilarity = np.zeros((self.trainset.n_items, self.trainset.n_items), dtype=np.double)
        tempEmpiricalDistributioni = {}
        tempEmpiricalDistributionj = {}
        Initial_Distribution = {}
        sum = 0
        for _, _, rate in self.trainset.all_ratings():
            if rate not in Initial_Distribution.keys():
                Initial_Distribution[rate] = 1
            else:
                Initial_Distribution[rate] += 1
            tempEmpiricalDistributioni[rate] = 0
            tempEmpiricalDistributionj[rate] = 0
            sum += 1
        tempEmpiricalDistributioni = OrderedDict(sorted(tempEmpiricalDistributioni.items()))
        tempEmpiricalDistributionj = OrderedDict(sorted(tempEmpiricalDistributionj.items()))
        Initial_Distribution = OrderedDict(sorted(Initial_Distribution.items()))
        position = []
        # print(Initial_Distribution)
        for rate in Initial_Distribution.keys():
            Initial_Distribution[rate] /= sum
            position.append(rate)
        # print('Initial_Distribution')
        # print(Initial_Distribution)
        # print(position)
        cnt = 0
        ave = 0
        for item, rates in self.trainset.ir.items():
            ave += len(rates)
            cnt += 1
        Average_RateNum = ave / cnt
        pbar = tqdm(total=(self.trainset.n_items * self.trainset.n_items))
        for itemi, ratesi in self.trainset.ir.items():
            for itemj, ratesj in self.trainset.ir.items():
                # if mySimilarity[itemj, itemi] != 0:
                #     mySimilarity[itemi, itemj] = mySimilarity[itemj, itemi]
                #     continue
                FinalDistri = []
                FinalDistrj = []
                for _, ratei in ratesi:
                    tempEmpiricalDistributioni[ratei] += 1
                for _, ratej in ratesj:
                    tempEmpiricalDistributionj[ratej] += 1
                for ratenum in sorted(Initial_Distribution):
                    # FinalDistri.append(
                    #     Initial_Distribution[ratenum] * Average_RateNum * beta + tempEmpiricalDistributioni[ratenum])
                    # FinalDistrj.append(
                    #     Initial_Distribution[ratenum] * Average_RateNum * beta + tempEmpiricalDistributionj[ratenum])
                    FinalDistri.append(
                        Initial_Distribution[ratenum] * self.bigC + tempEmpiricalDistributioni[ratenum])
                    FinalDistrj.append(
                        Initial_Distribution[ratenum] * self.bigC + tempEmpiricalDistributionj[ratenum])
                # mySimilarity[itemi, itemj] = (1 / (
                #             1 + wasserstein_distance(position, position, dirichlet.mean(FinalDistri),
                #                                      dirichlet.mean(FinalDistrj)))) ** gamma
                itemSimilarity[itemi, itemj] = np.exp(
                    (-1) * wasserstein_distance(position, position, dirichlet.mean(FinalDistri),
                                                dirichlet.mean(FinalDistrj)))
                # print(FinalDistri, FinalDistrj, itemSimilarity[itemi, itemj])
                for key in sorted(tempEmpiricalDistributioni):
                    tempEmpiricalDistributioni[key] = 0
                    tempEmpiricalDistributionj[key] = 0
                pbar.update(1)
        pbar.close()
        print('Done computing EMD similarity matrix.')

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemMaxlen = max([len(rates) for _, rates in self.trainset.ir.items()])
        self.userMaxlen = max([len(rates) for _, rates in self.trainset.ur.items()])

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                PSS1 = 1
                PSS2 = 0
                Commonitems = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            Commonitems += 1
                        Proximity = 1 - (1 + math.exp(-math.fabs(ratei - ratej))) ** -1
                        Significance = (1 + math.exp(
                            -1 * math.fabs(ratei - self.medianvalue) * math.fabs(ratej - self.medianvalue))) ** -1
                        Singularity = 1 - (1 + math.exp(-math.fabs(
                            0.5 * (ratei + ratej) - 0.5 * (self.itemmeans[itemi] + self.itemmeans[itemj])))) ** -1
                        Antipopularity = (2 - (1 + math.exp(-len(self.trainset.ir[itemi]) / self.itemMaxlen)) ** -1) * (
                                    2 - (1 + math.exp(-len(self.trainset.ir[itemj]) / self.itemMaxlen)) ** -1)
                        Antiprominent = (2 - (1 + math.exp(-len(self.trainset.ur[userj]) / self.userMaxlen)) ** -1)
                        if itemi == itemj:
                            PSS1 *= (1 + Antipopularity * Proximity * Significance * Singularity * Antiprominent * itemSimilarity[itemi, itemj])
                        else:
                            PSS2 += Antipopularity * Proximity * Significance * Singularity * Antiprominent * itemSimilarity[itemi, itemj]
                PSS2 = (1 + PSS2 / (len(ratesi) * len(ratesj) - Commonitems)) if (len(ratesi) * len(
                    ratesj) - Commonitems) != 0 else 1
                PSS = PSS1 * PSS2
                Jacci = (1 + math.exp((-1) * Commonitems / len(ratesi))) ** -1 if len(ratesi) != 0 else 0
                # Jacci = (1 + math.exp((-1) * (Commonitems / len(ratesi) + Commonitems / (len(ratesi) + len(ratesj) - Commonitems)))) ** -1 if len(ratesi) != 0 else 0
                # Jaccj = (1 + math.exp((-1) * Commonitems / len(ratesj))) ** -1 if len(ratesi) != 0 else 0
                # URP = 1 - (1 + math.exp((-1) * math.fabs(self.means[useri] - self.means[userj]) * math.fabs(
                #     self.std[useri] - self.std[userj]))) ** -1
                URP = (1 - (1 + math.exp((-1) * math.fabs(self.means[useri] - self.means[userj]))) ** -1) * (1 - (1 + math.exp((-1) * math.fabs(
                    self.std[useri] - self.std[userj]))) ** -1)
                # URP = (2 - (1 + math.exp((-1) * math.fabs(self.means[useri] - self.means[userj]))) ** -1) * (
                #             2 - (1 + math.exp((-1) * math.fabs(
                #         self.std[useri] - self.std[userj]))) ** -1)
                mySimilarity[useri, userj] = Jacci * URP * PSS
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
        # print(1)
        # print(mySimilarity)
        print(self.sim)
        print('Done computing user similarity matrix.')

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):  # both know
            raise PredictionImpossible('User and/or item is unknown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        # print(details)
        return est, details

    def nspCalc(self):
        sp = 0
        total = 0

        for user in self.testusers:
            for item in self.trainset.ir.keys():
                total += 1
                x, y = user, item
                neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
                if not neighbors:  # empty list
                    continue
                k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
                sum_sim = sum_ratings = actual_k = 0
                for (nb, sim, r) in k_neighbors:
                    if sim > 0:
                        sum_sim += sim
                        sum_ratings += sim * (r - self.means[nb])
                        actual_k += 1

                if actual_k < self.min_k:
                    continue
                if sum_sim == 0:
                    continue
                sp += 1
        return sp/total

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions


if __name__ == '__main__':
    from cdsds import CalMetric
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(diri, 5)
    mae, rmse, p, rec, f, npp, nsp = CalMetric().nocvcalculate(diri)
    print('#'*100)
    print('This is antipopularity model')
    print('#'*100)
    print('mae = ' + str(mae))
    print('rmse = ' + str(rmse))
    print('pre = ' + str(p))
    print('rec = ' + str(rec))
    print('f1 = ' + str(f))
    print('npp = ' + str(npp))
    print('nsp = ' + str(nsp))


    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(diri, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is my model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./pssaa.npy', resultsDict)
    # np.save('./pssaa2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is my model')
    # print('#'*100)
