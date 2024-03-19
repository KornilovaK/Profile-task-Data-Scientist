from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from features import init_features

data_root = 'Profile-task-Data-Scientist/data'
model_root = 'Profile-task-Data-Scientist/models'


def init():
    init_features()

    train_features = pd.read_csv(os.path.join(data_root, 'train_features.csv'))
    test_features = pd.read_csv(os.path.join(data_root, 'test_features.csv'))
    test_ids = pd.read_csv(os.path.join(data_root, 'test_ids.csv'))
    
    X, y = train_features.drop(["score", "Unnamed: 0"], axis=1), train_features["score"]
    X_train, X_val, y_train,  y_val = train_test_split(X, y, random_state=42, test_size=0.2)
    X_test, test_ids = test_features.drop(["Unnamed: 0"], axis=1), test_ids.drop(["Unnamed: 0"], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_lgb_reg(X_train, X_val, y_train,  y_val)
    predict(X_test, test_ids)


def train_lgb_reg(X_train, X_val, y_train,  y_val):
    model_lgb = LGBMRegressor(metric='rmse', random_state=42)
    model_lgb.fit(X_train, y_train)
    pred_lgbreg = model_lgb.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_lgbreg))
    
    model_lgb.booster_.save_model(os.path.join(model_root, 'lgbm_reg.txt'))
    
    rmse = train_lgb_reg(X_train, X_val, y_train,  y_val)

    with open(os.path.join(data_root, "rmse.txt")) as f:
            f.write(str(rmse))    
    

def predict(X_test, test_ids):
    model_lgb = lgb.Booster(model_file=os.path.join(model_root, 'lgbm_reg.txt'))
    preds = model_lgb.predict(X_test)
    pd.DataFrame({'id': test_ids.id, 'score': preds}).to_csv(os.path.join(data_root, "submission.csv"))


if __name__ == '__main__':
	init()