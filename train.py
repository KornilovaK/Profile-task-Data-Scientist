
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import catboost as cb


data_root = 'data'
model_root = 'model'

train_features = pd.read_csv(os.path.join(data_root, 'train_features.csv'))
test_features = pd.read_csv(os.path.join(data_root, 'test_features.csv'))
test_ids = pd.read_csv(os.path.join(data_root, 'test_ids.csv'))


X, y = train_features.drop(["score", "Unnamed: 0"], axis=1), train_features["score"]
X_train, X_val, y_train,  y_val = train_test_split(X, y, random_state=42, test_size=0.2)

X_test, test_ids = test_features.drop(["Unnamed: 0"], axis=1), test_ids.drop(["Unnamed: 0"], axis=1)


params = {'learning_rate': 0.008,
          'depth': 8,
          'subsample': 0.3,
          'colsample_bylevel': 0.82,
          'min_data_in_leaf': 15,
          "iterations": 1000,
          "random_state": 42,
          "eval_metric": "RMSE"
          }

model = cb.CatBoostRegressor(**params)
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
model.save_model(os.path.join(model_root, "catboost.cbm"), format="cbm")

predictions = model.predict(X_val)
rmse = mean_squared_error(y_val, predictions, squared=False)

with open(os.path.join(data_root, "preiction_score.txt")) as f:
    f.write(str(rmse))

preds = model.predict(X_test)
pd.DataFrame({'id': test_ids.id, 'score': preds}).to_csv(os.path.join(data_root, "submission.csv"))