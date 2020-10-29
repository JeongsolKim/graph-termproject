import matplotlib.pyplot as plt
import numpy as np


dict = {}
for i in range(283):
    with open('valid_answer/valid_answer_'+str('%03d' %i) +'.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split('\t')[-1].split('\n')[0]
            if dict.get(sp):
                dict[sp] +=1
            else:
                dict[sp] = 1


def f1(x):
    return x[0]

sort_dict = sorted(dict.items(), key=f1, reverse=False)
print(sort_dict)

with open('valid_attacks.txt', 'w') as f:
    for item in sort_dict:
        f.write(item[0]+'\n')
