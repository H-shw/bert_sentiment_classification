# -- coding:UTF-8 --

import pandas
import jieba

# parameters
src_file = ['src/nCoV_100k_train.labled.csv', 'src/nCov_10k_test.csv']
res_raw_path = ['res/raw/train.csv', 'res/raw/test.csv']
res_path = ['res/fin/train.csv', 'res/fin/test.csv']
stopwords_path = 'res/raw/hit_stopwords.txt'
div_path = ['res/div/train.csv', 'res/div/valid.csv', 'res/div/test.csv']


def csv_rm():
    '''
    操作对象： 100k train data & 10k predict data
    做法: 1. 提取必要的数据 2.删除空数据(对于predict也做了这一步，导致后来还需要打补丁)
    :return: 初步数据(raw)
    '''
    df = pandas.read_csv(src_file[0], encoding='utf8')
    df = df.loc[:, ['微博id', '微博中文内容', '情感倾向']]
    df.columns = ['id', 'text', 'label']
    df = df.dropna()
    df.to_csv(res_raw_path[0], index=None)

    df = pandas.read_csv(src_file[1], encoding='utf8')
    df = df.loc[:, ['微博id', '微博中文内容']]
    df.columns = ['id', 'text']
    df = df.dropna()
    df.to_csv(res_raw_path[1], index=None)


def data_analysis():
    '''
    用来分析标签分布的，这里通过这一步可以发现一些标签数量非常稀少(只有1个)，或者标注可能有误？
    后面通过手动的操作删除了这些标签有误的数据
    :return:
    '''
    df = pandas.read_csv(div_path[1])
    print(df['label'].value_counts())

    '''
    分布情况：
    0     57619
    1     25392
    -1    16903
    '''


def cut_and_stop():
    '''
    text 分词，删除停用词(hit-stop-words)，然后再删除 text 为空的数据
    最后又将 text项 按照 str 存储，使得与后续的 BERT 输入要求符合
    对于 train predict数据都做了这个操作
    :return:
    '''
    stopwords = open(stopwords_path, "r", encoding='utf-8').read().split('\n')
    count = 0
    df = pandas.read_csv(res_raw_path[0])
    df2 = pandas.DataFrame(columns=['id', 'text', 'label'])
    for index, row in df.iterrows():
        words = list(jieba.cut(row['text']))
        words = list(set(words))
        words = [tmp for tmp in words if tmp not in stopwords]
        words = ' '.join(words)
        df2 = df2.append({'id': row['id'], 'text': words, 'label': row['label']}, ignore_index=True)
        count += 1
        print(count)
    df2.to_csv(res_path[0], index=False)

    count = 0
    df = pandas.read_csv(res_raw_path[1])
    df2 = pandas.DataFrame(columns=['id', 'text'])
    for index, row in df.iterrows():
        words = list(jieba.cut(row['text']))
        words = list(set(words))
        words = [tmp for tmp in words if tmp not in stopwords]
        words = ' '.join(words)
        df2 = df2.append({'id': row['id'], 'text': words}, ignore_index=True)
        count += 1
        print(count)
    df2.to_csv(res_path[1], index=False)


def tgt_analysis():
    '''
    1.
        分析经过删除停用词之后句子  token 的数量
        主要用来匹配 BERT 要求输入 <512 的条件
        结果是不存在 >512 的数据
        微博限制字数为 140
    2.
        删除 train 数据集当中 text 相同的数据

    :return:
    '''
    cnt1 = 0
    cnt2 = 0
    df = pandas.read_csv(res_path[0])
    df = df.dropna()
    for index, row in df.iterrows():
        # print(row['text'])
        # print(len(row['text']))
        try:
            if len(row['text'].split(' ')) > 512:
                cnt1 += 1
        except Exception:
            print(row['id'], row['text'])

    df = pandas.read_csv(res_path[1])
    df = df.dropna()
    for index, row in df.iterrows():
        if len(row['text'].split(' ')) > 512:
            cnt2 += 1

    print(cnt1)
    print(cnt2)
    '''
    超出512 ：0,0
    '''

    # 去重 (text相同)
    df = pandas.read_csv(res_path[0])
    df = df.dropna()
    df.drop_duplicates(subset=['text'], keep=False, inplace=True)
    df.to_csv(res_path[0], index=False)


def data_div():
    '''
    切分 train 数据集
    按照 8:1:1 切分为 训练集，验证集，测试集
    随机打乱数据
    :return:
    '''
    df = pandas.read_csv(res_path[0])
    train_data = df.sample(frac=0.8, axis=0)
    temp_data = df[~df.index.isin(train_data.index)]
    valid_data = temp_data.sample(frac=0.5, axis=0)
    test_data = temp_data[~temp_data.index.isin(valid_data.index)]
    # print(len(df),len(train_data),len(valid_data),len(test_data))
    train_data.to_csv(div_path[0], index=False)
    valid_data.to_csv(div_path[1], index=False)
    test_data.to_csv(div_path[2], index=False)


if __name__ == '__main__':
    '''
    全流程：
    1.提出必要的信息
    2.分析标签分布并处理
    3.删去停用词
    4.分析句子长度和去除重复的数据
    5.划分 train 数据集 为 train/dev/test
    '''
    # csv_rm()
    data_analysis()
    # cut_and_stop()
    # tgt_analysis()
    # data_div()
    pass
