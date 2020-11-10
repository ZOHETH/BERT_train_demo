import pandas as pd
import pickle
import numpy as np
from bert_serving.client import BertClient

TAG_PATH = 'dataset/tag.csv'
TRAIN_DATA_PATH = 'dataset/data_t.npy'
TAG_COLUMN_NAME = '类别'


def load_tag():
    df = pd.read_csv('tag_data.csv')
    cur = ''
    for i, row in df.iterrows():
        if row['数量'] < 0:
            cur = row['类别']
        else:
            df.loc[i, '类别'] = cur + '.' + row['类别']
    df = df[df['数量'] >= 100]['类别']
    df.to_csv('tag.csv', index=False)


def tag_sum(x, tags):
    res = ''
    for tag in tags:
        res += x[tag] + '.'
    return res[:-1]


class BertEncoding:
    def __init__(self):
        self.data_df = None
        self.tags_dict = {}
        self.bert_client = BertClient()

    def file_tag(self):
        df = pd.read_csv(TAG_PATH)
        for i, row in df.iterrows():
            self.tags_dict[row[TAG_COLUMN_NAME]] = i

    def load_data(self, data_df, tag_columns: tuple, text_column, load_tag_from_file=True, save_data=True):
        df = data_df
        # 多种层次的类别化为一个类别
        df[tag_columns[-1]] = df.apply(lambda x: tag_sum(x, tag_columns), axis=1)
        # 为每个类别编号
        if load_tag_from_file:
            self.file_tag()
        else:
            tags = df[tag_columns[-1]].unique()
            for i, tag in enumerate(tags):
                self.tags_dict[tag] = i
        # 编码
        data = self.encoding(df, tag_columns[-1], text_column)
        if save_data:
            np.save(TRAIN_DATA_PATH, data)
        return data

    def encoding(self, data_df, tag_column, text_column):
        df = data_df
        # 将类别转换成数字
        df = df[df[tag_column].isin(self.tags_dict)]
        df[tag_column] = df[tag_column].transform(lambda x: self.tags_dict[x])
        # 去除空值 无效值等
        df = df.loc[(~df[text_column].isnull()) & (df[text_column].str.len() > 2)]
        df = df.reset_index(drop=True)

        print('总数据: {}'.format(len(df)))
        encodes = self.bert_client.encode(list(df[text_column].values))
        print(encodes.shape, df[tag_column].values.T[:, np.newaxis].shape)
        data = np.hstack((encodes, df[tag_column].values.T[:, np.newaxis]))

        # df['问'] = df['问'].transform(lambda x: bert_client.encode([x]))
        print(data.shape)
        # np.save('data_t.npy', data)
        return data

    def load_new_data(self, tag_names: tuple = None, old_npy_path=TRAIN_DATA_PATH):
        df = pd.read_csv('dataset/data.csv')
        df = df.loc[df['大类'] == '拒绝购买']
        df = df.loc[df['小类'].isin(('其他', '已经有很多保险'))]
        new_npy = self.load_data(df, ('大类', '小类'), '问', save_data=False)
        old_npy = np.load(old_npy_path)
        new_npy = np.vstack((old_npy, new_npy))
        np.save(old_npy_path, new_npy)


if __name__ == '__main__':
    be = BertEncoding()
    be.load_new_data()
