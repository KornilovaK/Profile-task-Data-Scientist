import pandas as pd
import numpy as np
import os
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def init():
    root = 'data'
    features = pd.read_csv(os.path.join(root, 'features.csv'))
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    prepare_df(train, test, features)


def prepare_df(train, test, features):
    dfs = [train, test, features]

    for l in ['lat', 'lon']:
        for df in dfs:
            df[l] = round(df[l], 6)

    for df in dfs:
        df["coord"] = round(df.lat * df.lon, 1)

    features["id"] = list(range(0, features.shape[0]))
    


def 
    feat_cols = features.columns.values[2:-2]
    coords_f = features.coord.values
    coords_tr = train.coord.values
    coords_ts = test.coord.values

    for f in feat_cols:
        train[f] = np.zeros(train.shape[0])
        test[f] = np.zeros(test.shape[0])

def fill_df(features, df):

    for i in range(len(coords_f)):
        f_i = coords_f[i]

        if (f_i in coords_tr) or ((f_i+0.1) in coords_tr) or ((f_i-0.1) in coords_tr):
            if f_i in coords_tr: js = np.where (coords_tr == f_i)
            elif (f_i+0.1) in coords_tr: js = np.where (coords_tr == (f_i+0.1))
            elif (f_i-0.1) in coords_tr: js = np.where (coords_tr == (f_i-0.1))

            for j in js:
                for id, f in enumerate(features.loc[i][2:-2].values):
                    train.loc[j, str(id)] = f

    for i in range(train.shape[0]):
        if df .loc[i, feat_cols].sum() == 0:
            df = train.drop(i)


train.drop(["id"], axis=1).to_csv(os.path.join(root, "train_features.csv"))
pd.DataFrame({"id": train["id"]}).to_csv(os.path.join(root, "train_ids.csv"))


for i in ranage(len(coords_f)):
  f_i = coords_f[i]

  if (f_i in coords_ts) or ((f_i+0.1) in coords_ts) or ((f_i-0.1) in coords_ts):
    if f_i in coords_ts: js = np.where (coords_ts == f_i)
    elif (f_i+0.1) in coords_ts: js = np.where (coords_ts == (f_i+0.1))
    elif (f_i-0.1) in coords_ts: js = np.where (coords_ts == (f_i-0.1))

    for j in js:
      for id, f in enumerate(features.loc[i][2:-2].values):
        test.loc[j, str(id)] = f

for i in range(test.shape[0]):
  if test.loc[i, feat_cols].sum() == 0:
    test = test.drop(i)

test.drop(["id"], axis=1).to_csv(os.path.join(root, "test_features.csv"))
pd.DataFrame({"id": test["id"]}).to_csv(os.path.join(root, "test_ids.csv"))

if __name__ == '__main__':
	init()