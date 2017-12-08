
# coding: utf-8

# In[13]:


# CS454 Coursework #2
# 20130143 Yihan Kim

# 내장 모듈(안 쓸 수도 있음)
import os
import sys
import csv
import math
import array
import pickle
import random
import operator
import itertools

# 외부 모듈(안 쓸 수도 있음)
import numpy 
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# In[14]:


# 전역 변수

# 사용 가능한 데이터 파일들의 리스트
file_list = os.listdir("fluccs_data")

# 맨 앞의 method, 맨 뒤의 fault는 학습 데이터로 사용할 수 없음
# 아래 데이터는 0에서 1 사이의 normalized된 값들
data_pivot = ['ochiai', 'jaccard', 'gp13', 'wong1', 'wong2', 
              'wong3', 'tarantula', 'ample', 'RussellRao', 'SorensenDice', 
              'Kulczynski1', 'SimpleMatching', 'M1', 'RogersTanimoto', 
              'Hamming', 'ochiai2', 'Hamann', 'dice', 'Kulczynski2', 
              'Sokal', 'M2', 'Goodman', 'Euclid', 'Anderberg', 'Zoltar', 
              'ER1a', 'ER1b', 'ER5a', 'ER5b', 'ER5c', 'gp02', 'gp03', 
              'gp19', 'min_age', 'max_age', 'mean_age', 'num_args', 
              'num_vars', 'b_length', 'loc', 'churn']

num_column = len(data_pivot) # == 41

# fitness 함수
# data_row ->> float
fitness_f = None

# data, label의 집합
# 2D array로 사용 : datas[index_of_data_set][row_index]
datas = []
labels = []

rankings = []


# In[15]:


def acquire_csv(filename):
    # 실제로 사용하는 데이터 형식
    # data : fault 컬럼을 제외한 나머지 
    # label : fault 컬럼
    # fault 컬럼을 학습에 직접 사용하면 안된다고 했으므로
    # 만일의 사태를 대비하여 별도로 관리한다
    
    def _acquire_raw_csv(filename):
        # fluccs_data 폴더 안의 데이터를 리스트로 읽어서 리턴
        # index row는 포함하지 않음
        assert filename in file_list, "File does not exist in fluccs_data/."
        f = None

        with open("fluccs_data/" + filename, newline="") as csvfile:
            f = list(csv.reader(csvfile, delimiter=",", quotechar="|"))
            assert f != None, "Failed to read data"
            assert f[0][1:-1] == data_pivot, f[0]

        return f[1:]
    
    raw_data = _acquire_raw_csv(filename)
    data = list()
    label = list()
    
    for data_line in raw_data:
        data.append(list(map(float, data_line[1:-1])))
        label.append(int(data_line[-1]))
    
    return data, label

# example
#d, l = acquire_csv("Math_1.csv")


# In[23]:


def read_data(file_names):
    # train / validate 할 여러 개의 데이터를 받아다가 저장함 
    #print("reading files...")
    global datas, labels
    datas = []
    labels = []
    
    for filename in file_names:
        #print("reading " + filename)
        data, label = acquire_csv(filename)
        datas.append(data)
        labels.append(label)
        
    #print("reading finished.")
    return 

# example
#read_data(file_list[:120])


# In[24]:


'''
read_data(file_list[140:])

f = lambda x: sum([x[i] * 1 for i, v in enumerate(x)])
z = eval_function(f)
print (len(z))
print (len(list(filter(lambda x: x < 1, z))))
'''


# In[25]:


# 아래 symbolic regression 코드는 DEAP example에서 받아옴.

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 41)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
try:
    pset.addEphemeralConstant("rand101", lambda: random.random())
except:
    True

# 수정
for idx, name in enumerate(data_pivot):
    command = "pset.renameArguments(ARG%s='%s')" % (str(idx), name)
    exec (command)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -10.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#추가
def eval_function(f):
    # 함수의 evaluation & ranking이 얼마나 정확한지
    # y : 랭킹의 합
    global rankings
    rankings = []
    
    fitness = list()
    for idx in range(len(datas)):
        data, label = datas[idx], labels[idx]
        score_list = enumerate(map(lambda x: f(*x), data))
        score_list = sorted(score_list, key=lambda x:x[1], reverse=True)
        ranking = list(enumerate(map(lambda x: x[0], score_list)))
        rankings.append(ranking[0][1])
        fault = [rank for rank, idx in ranking if label[idx] == 1]
        fitness.append(sum(fault) / len(fault))

    return sum(fitness) / len(fitness), min(fitness)
# example
# f = lambda x: sum([x[i] * 1 for i, v in enumerate(x)])
# z = eval_function(f)
# print (len(z))
# print (len(list(filter(lambda x: x < 1, z))))

# 수정
def evalSymbReg(individual):
    func = toolbox.compile(expr=individual)
    return (eval_function(func))

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))



# In[26]:


def training():
    random.seed()

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 20, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


# In[27]:


def cross_validate():
    pivot = (4 * len(file_list)) // 10
    file_list_ = random.sample(file_list, pivot)
    # training data : 
    read_data(file_list_)
    pop, log, hof = training()
    pickle.dump(hof[0], open('model.pkl', 'wb'))
    validate_list_ = list(set(file_list) - set(file_list_))
    read_data(validate_list_)
    print(evalSymbReg(hof[0]))
    return 


# In[28]:


def main():
    if len(sys.argv) == 1:
        cross_validate()
    else:
        if sys.argv[1] == "train":
            read_data(sys.argv[2:])
            pop, log, hof = training()
            pickle.dump(hof[0], open('model.pkl', 'wb'))
        elif sys.argv[1] == "validate":
            f = open("model.pkl","rb")
            bin_data = f.read()
            model = pickle.loads(bin_data)
            read_data(sys.argv[2:])
            evalSymbReg(model)
            for i in rankings:
                print(i)
        else:
            pass
        
if __name__ == "__main__":
    main()


