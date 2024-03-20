from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from features import init_features


data_root = 'Profile-task-Data-Scientist/data'
model_root = 'Profile-task-Data-Scientist/models'


def prepare_data(train_features: pd.DataFrame,
                 test_features: pd.DataFrame,
                 test_ids: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame,
                 np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:

    X, y = train_features.drop(["score", "Unnamed: 0"], axis=1), train_features["score"]
    X_train, X_val, y_train,  y_val = train_test_split(X, y, random_state=42, test_size=0.2)
    X_test, test_ids = test_features.drop(["Unnamed: 0"], axis=1), test_ids.drop(["Unnamed: 0"], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, test_ids


def init() -> None:
    init_features(data_root)

    try:
        train_features = pd.read_csv(os.path.join(data_root, 'train_features.csv'))
        test_features = pd.read_csv(os.path.join(data_root, 'test_features.csv'))
        test_ids = pd.read_csv(os.path.join(data_root, 'test_ids.csv'))
    except:
        init_features(data_root)
    
    
    X_train, y_train, X_val, y_val, \
             X_test, test_ids = prepare_data(train_features,
                                             test_features, test_ids
                                            )

    train_lgb_reg(X_train, X_val, y_train, y_val)
    train_lgb(X_train, X_val, y_train, y_val)
    predict(X_val, y_val, X_test, test_ids)


def train_lgb_reg(X_train: np.ndarray, X_val: np.ndarray,
                  y_train: pd.DataFrame,  y_val: pd.DataFrame) -> None:

    model_lgb = lgb.LGBMRegressor(metric='rmse', random_state=42)
    model_lgb.fit(X_train, y_train)
    pred_lgbreg = model_lgb.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_lgbreg))
    
    model_lgb.booster_.save_model(os.path.join(model_root, 'lgbm_reg.txt'))
    
    with open(os.path.join(data_root, "rmse.txt"), 'w') as f:
        f.write("LGBMRegressor: " + str(rmse))
    

def train_lgb(X_train: np.ndarray, X_val: np.ndarray,
              y_train: pd.DataFrame, y_val: pd.DataFrame) -> None:

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            "random_state": 42
        }

    num_round = 100
    bst = lgb.train(params, train_data,
                    num_round, valid_sets=[val_data])

    pred_lgb = bst.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_lgb))
    bst.save_model(os.path.join(model_root, 'lgbm.txt'), num_iteration=bst.best_iteration)

    with open(os.path.join(data_root, "rmse.txt"), 'a') as f:
        f.write("\nLGBM: " + str(rmse))


def predict(X_val: np.ndarray, y_val: pd.DataFrame,
            X_test: np.ndarray, test_ids: pd.DataFrame) -> None:

    model_lgb = lgb.Booster(model_file=os.path.join(model_root, 'lgbm_reg.txt'))
    bst = lgb.Booster(model_file=os.path.join(model_root, 'lgbm.txt'))

    lgb_pred_val = model_lgb.predict(X_val)
    bst_pred_val = bst.predict(X_val)

    preds_val = (lgb_pred_val + bst_pred_val) / 2
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))

    with open(os.path.join(data_root, "rmse.txt"), 'a') as f:
        f.write("\nEnsemle: " + str(rmse))

    lgb_pred = model_lgb.predict(X_test)
    bst_pred = bst.predict(X_test)

    preds = (lgb_pred + bst_pred) / 2

    pd.DataFrame({'id': test_ids.id, 'score': preds}).to_csv(os.path.join(data_root, "submission.csv"))


if __name__ == '__main__':
	init()