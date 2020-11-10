import numpy as np
import pandas as pd
import pickle
import io
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from bert_serving.client import BertClient


TAG_COLUMN_NAME = '类别'
TAG_PATH = 'dataset/tag.csv'

class Evaluate:
    def __init__(self, model: LogisticRegression = None, ):
        self.model = model
        self.tags_list = []
        self.tags_dict = {}
        self.need_set = set()
        self.load_tag()
        self.tags_df = pd.read_csv(TAG_PATH)

    def load_tag(self):
        df = pd.read_csv(TAG_PATH)
        for i, row in df.iterrows():
            self.tags_dict[row[TAG_COLUMN_NAME]] = i
            self.tags_list.append(row[TAG_COLUMN_NAME])
            if row['need'] == 1:
                self.need_set.add(i)

    def show_need_scores(self, y_true, y_pred):
        y_t = pd.Series(list(y_true))
        y_p = pd.Series(list(y_pred))
        df_t = pd.DataFrame({'true': y_t, 'pred': y_p})

        # temp = np.equal(y_true, y_pred)
        # mat = np.vstack((y_true, y_pred, temp)).T
        # groups=npi.group_by(mat[:, 0]).split(mat[:, 1:])
        # a=1
        tp = df_t.loc[df_t['true'] == df_t['pred']]['true'].value_counts()
        acc = tp / df_t['pred'].value_counts()
        acc.loc['sum'] = len(df_t.loc[df_t['true'] == df_t['pred']]) / len(df_t.loc[~df_t['pred'].isnull()])

        recall = tp / df_t['true'].value_counts()
        recall.loc['sum'] = len(df_t.loc[df_t['true'] == df_t['pred']]) / len(df_t.loc[~df_t['true'].isnull()])

        # acc = acc.rename({'pred': 'Acc'}, inplace=True)
        # recall = recall.rename({'true': 'Recall'}, inplace=True)
        # tp = tp.rename({'true': 'TP'}, inplace=True)
        res = pd.concat([self.tags_df, acc, recall, tp], axis=1)
        res.to_csv('res.csv', index=False)

        print('Acc:\n', acc, '\n\nRecall:\n', recall)
        return acc, recall

    def show_scores(self, x, y_true, model: LogisticRegression):
        y_pred = model.predict_proba(x)

    def load_trash_data(self):
        df = pd.read_csv('data.csv')
        df = df.loc[df['小类'] == '无意义']
        bert_client = BertClient()


if __name__ == '__main__':
    lr = joblib.load('model.pkl')
    e = Evaluate(lr)
