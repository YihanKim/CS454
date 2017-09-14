#! /usr/bin/env python3
# 우생학적으로 패스워드 맞추기

import random

# 1. 상수들
# domain of symbols
digit_dom = list(map(str, range(10)))
answer = "151829348903218590843290675532152313"
pw_length = len(answer)
sample_size = len(answer) * 7

# 2. 함수들
# verify : return correct # of digits
def verify(s):
    return len([1 for i in range(pw_length) if s[i] == answer[i]])

# mix : return randomly mixed string of two password strings
def mix(s, t):
    return "".join([s[i] if random.random() > 0.5 else t[i] for i in range(pw_length)])

# 3. 샘플 생성
x = ["".join([digit_dom[random.randint(0, 9)] for i in range(pw_length)]) for j in range(sample_size)]
x = sorted(x, key=lambda x: -verify(x))[:sample_size]
print(x[0], verify(x[0]))

# 4. 진화
while True:
    for j in range(sample_size):
        a = random.randint(0, sample_size - 1);
        b = random.randint(0, sample_size - 1);
        x.append(mix(x[a], x[b]))
    x = sorted(x, key=lambda x: -verify(x))[:sample_size]
    print(x[0], verify(x[0]))
    if verify(x[0]) == len(answer): break
