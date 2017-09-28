 
# coding: utf-8

# In[2]:


#! /usr/bin/env python3
# 20130143 Yihan Kim

# 1. 모듈 불러오기

# 파이썬 라이브러리
import os
import math
import string
import random
import time
import datetime
import multiprocessing.process

# 외부 라이브러리
import numpy as np


# In[3]:


# 2. 파일 입출력

# ./data 폴더의 tsp 파일을 파싱
# data 폴더 중에서 tsp 확장자가 아닌 파일은 에러 처리
def get_filedirs():
    path = os.path.join(os.getcwd(), "data")
    if os.path.isdir(path):
        return list(map(lambda fname: os.path.join(path, fname), os.listdir("data")))
    return list()

# N * 2 Array로 읽어들이는 과정 - 11849에서 0.05초 가량 소요
def get_contents(fpath):
    extension = fpath.split(".")[-1]
    assert extension in ["tsp", "TSP"], "Unable to read : file extension %s is not supported" % extension
    with open(fpath) as f:
        return [[int(x[0]), np.array([float(x[1]), float(x[2])])] for x in [l.split() for l in f if l[0] in string.digits]]


# In[4]:


# 3. 기본 함수들
# 두 개의 점을 입력받아 두 점 사이의 거리를 리턴
def get_distance(i, j):
    return np.linalg.norm(i[1] - j[1])

# 0부터 n-1까지를 포함한 arrangement 리턴
def gen_random(n):
    return np.random.permutation(range(n))

# permutation과 tsp data를 바탕으로 총 길이를 evaluate
def get_total_length(tsp_data, permutation):
    length = len(tsp_data)
    distance = 0;
    for i in range(length):
        distance += get_distance(tsp_data[permutation[i - 1]], tsp_data[permutation[i]])
    return [distance]


# In[5]:


# 4. 진화 알고리즘 요소
def inverse(l):
    # rev_l에 값으로 접근하면 index의 값을 얻을 수 있음
    rev_l = [0] + [0 for i in l]
    for k, v in enumerate(l):
        rev_l[v] = k 
    return rev_l

def swap(l, a, b):
    l[a], l[b] = l[b], l[a]

# permutation의 PMX 크로스오버 사용
def crossover(x, y, cprob):
    assert len(x) == len(y)
    x = list(x)
    y = list(y)
    n = len(x)
    crosslength = int((1 - cprob) * n)
    initialidx = random.randint(0, n - crosslength)
    x_, y_ = x[:], y[:]
    xinv = inverse(x)
    yinv = inverse(y)
    z = [-1 for i in range(n)]
    w = [-1 for i in range(n)]
    interval = sorted([initialidx, crosslength + initialidx])
    for i in range(*interval):
        value = y[i]
        x_idx = xinv[value]
        swap(x_, i, x_idx)
        value = x[i]
        y_idx = yinv[value]
        swap(y_, i, y_idx)
        
    return (x_, y_)


# x의 요소들을 mutate_prob의 확률로 변화(swap)시키는 함수
def mutate(x, mutate_prob):
    for i in range(int(mutate_prob * len(x))):
        j = random.randint(0, len(x) - 1)
        k = random.randint(0, len(x) - 1)
        if j != k:
            swap(x, j, k)
    return
    
 


# In[6]:


# 5. 결정론적 알고리즘
# Optimal solution의 2배까지 나오는 결과물
def tsp_nn(tsp_data):
    k = random.randint(0, len(tsp_data) - 1)
    d = tsp_data[:]
    n = [tsp_data[k]]
    d.pop(k)
    
    while len(d) > 0:
        min_dist = 1e10
        idx = -1
        for (i, j) in enumerate(d):
            new_dist = get_distance(n[-1], j)
            if new_dist < min_dist:
                min_dist, idx = new_dist, i
        n.append(d[idx])
        d.pop(idx)
    return n


# In[7]:


# 6. 상수
cprob = 0.95 # crossover probability
mprob = 0.02 # mutation probability
population = 40 # test case
elitism = 3 # number of elements will be preserved
maxgen = 10 # number of generations


# In[8]:


# 7. 유전 알고리즘

def main_ga():
    for filedir in get_filedirs():
        try:
            tsp_data = get_contents(filedir)
        except:
            continue
        
        d = {}
        def get_length_cache(p):
            try:
                return d[id(p)]
            except KeyError:
                d[id(p)] = get_total_length(tsp_data, p)
                return d[id(p)]
            
        tsp_length = len(tsp_data)
        instances = []
        
        for i in range(population):
            instances.append(gen_random(tsp_length))
            
        for gen in range(maxgen):
            instances.sort(key = lambda p: get_length_cache(p))
            instances = instances[:population]
            print("min distance in gen %s : %s" % (gen, get_total_length(tsp_data,instances[0])))
            nextgen = instances[:elitism]
        
            for _ in range(population - elitism):
                i = random.randint(0, population - 1)
                j = random.randint(0, population - 1)
                r = crossover(instances[i], instances[j], cprob)
                instances.append(r[0])
                instances.append(r[1])

            for x in instances:
                mutate(x, mprob)

# In[ ]:


# 8. 탐욕법 알고리즘
def main_nn():
    for filedir in get_filedirs():
        try:
            tsp_data = get_contents(filedir)
        except:
            continue
        tsp_length = len(tsp_data)
        p = gen_random(tsp_length)
        dat = tsp_nn(tsp_data)
        path = os.path.join(os.getcwd(), "result/%s_%s.txt" % (filedir.split("/")[-1], str(datetime.datetime.now())))
        print("%s : %s" % (filedir.split("/")[-1], get_total_length(dat, range(tsp_length))))
        with open(path, "w") as f:
            for i in dat:
                f.write(str(i[0]) + "\n")


# In[ ]:


if __name__ == "__main__":
    main_nn()


