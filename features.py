import pandas as pd
import numpy as np
import os
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def init_features(root: str) -> None:
    features = pd.read_csv(os.path.join(root, 'features.csv'))
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))

    feat_cols = features.columns.values[2:-2]
    train, test, features = prepare_df(train, test, features, feat_cols)

    train = fill_df(features, train, feat_cols)
    test = fill_df(features, test, feat_cols)

    save_new_data(root, train, test)


def prepare_df(train: pd.DataFrame, test: pd.DataFrame, features: pd.DataFrame,
               feat_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    dfs = [train, test, features]

    for l in ['lat', 'lon']:
        for df in dfs:
            df[l] = round(df[l], 6)

    for df in dfs:
        df["coord"] = round(df.lat * df.lon, 1)

    features["id"] = list(range(0, features.shape[0]))
    

    for f in feat_cols:
        train[f] = np.zeros(train.shape[0])
        test[f] = np.zeros(test.shape[0])

    return train, test, features


def fill_df(features: pd.DataFrame, df: pd.DataFrame,
            feat_cols: list) -> pd.DataFrame:
    print(df)
    all_js = set()
    coords_f = features.coord.values
    coords = df.coord.values
    len_f = len(coords_f)
    print(len_f)

    for i in range(len_f):
        f_i = coords_f[i]
        print(i, i/len_f)

        if (f_i in coords) or ((f_i+0.1) in coords) or ((f_i-0.1) in coords):
            if f_i in coords: js = np.where (coords == f_i)
            elif (f_i+0.1) in coords: js = np.where (coords == (f_i+0.1))
            elif (f_i-0.1) in coords: js = np.where (coords == (f_i-0.1))

            for j in js[0]:
                if j not in all_js:
                    for id, f in enumerate(features.loc[i][2:-2].values):
                        df.loc[j, str(id)] = f
                    all_js.add(j)

    print("all")
    for i in range(df.shape[0]):
        if df.loc[i, feat_cols].sum() == 0:
            df = df.drop(i)
    print("all2")
    
    return df


def save_new_data(root: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    train.drop(["id"], axis=1).to_csv(os.path.join(root, "train_features.csv"))

    test.drop(["id"], axis=1).to_csv(os.path.join(root, "test_features.csv"))
    pd.DataFrame({"id": test["id"]}).to_csv(os.path.join(root, "test_ids.csv"))