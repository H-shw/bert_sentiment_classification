# -- coding:UTF-8 --

'''
一个补丁文件
1. 用来修正一些在 predict 数据集当中在 prepocess 过程中被意外除去的数据（少了37条），给这些数据设定预测标签为0
2. 保证顺序与提交样本相同
'''

import pandas as pd
import random
import csv

src = 'epoch3batch60/ans.csv'
tgt = 'epoch3batch60/ans_patch.csv'
ref = '../dataset/src/submit_example.csv'


def ans_patch():
    df = pd.read_csv(ref)
    res = {}

    csvFileObj = open(src)
    readerObj = csv.reader(csvFileObj)
    for row in readerObj:
        if readerObj.line_num == 1:
            continue
        res[row[0]] = row[1]

    f = open(tgt, 'w', encoding='utf8')
    f.write('id,y\n')
    count = 0
    # print(res)
    for index, row in df.iterrows():
        # print(row[0],type(row[0]))

        if str(row[0]) in res.keys():
            f.write(f'{row[0]} ,{res[str(row[0])]}\n')
            # pass
        else:
            f.write(f'{row[0]},0\n')
            print(count)
            count += 1
    f.close()


if __name__ == '__main__':
    ans_patch()
