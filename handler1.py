from aassp import diri
import numpy as np
import time
from cdsds import CalMetric
import timeit


resultsDict = {}
nppnsp = {}
neighbours0 = [10, 20, 30, 40, 50, 60]
neighbours = [20, 40, 60, 80, 100, 120]
neighbours2 = [40, 80, 120, 160, 200, 240]
neighbours3 = [10, 40, 70, 100, 130, 160]
neighbours53 = [20, 40, 60, 80, 100, 120, 140, 180, 220, 260, 300, 340, 400, 460, 520, 580, 640, 700]
neighbourst1 = [10, 40, 70, 100, 130, 160]

start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(diri, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is my model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./pssaa.npy', resultsDict)
# np.save('./pssaa2.npy', nppnsp)
np.save('./pssaa_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is my model')
print('#' * 100)
