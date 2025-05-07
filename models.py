import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mlflow.tracking import MlflowClient
import warnings

warnings.filterwarnings("ignore")

# Veri setini yükle
df = pd.read_csv("Mobiles Dataset (2025).csv", encoding='latin1')
df.columns = df.columns.str.strip()

# Hedef değişkeni temizle
df['Launched Price (USA)'] = (
    df['Launched Price (USA)']
    .astype(str)
    .str.replace('USD', '', regex=False)
    .str.replace(',', '')
    .str.strip()
    .astype(float)
)

# Veriyi ikiye ayır (simüle edilmiş veri versiyonları)
split_index = int(len(df) * 0.5)
df_v1 = df.iloc[:split_index].copy()
df_v2 = df.iloc[split_index:].copy()

# df_v1'i train ve val olarak ayır
X = df_v1.drop(columns=['Launched Price (USA)']).dropna()
y = df_v1.loc[X.index, 'Launched Price (USA)']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# df_v2 test seti
X_final = df_v2.drop(columns=['Launched Price (USA)']).dropna()
y_final = df_v2.loc[X_final.index, 'Launched Price (USA)']

# Sayısal ve kategorik sütunlar
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Ön işleme
preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# MLflow ayarları
client = MlflowClient()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment=mlflow.set_experiment("Mobile Prices Pred")

# Dataset versiyonlarını logla
mlflow_train_dataset = mlflow.data.from_pandas(X_train.join(y_train), source="train", name="train_dataset")
mlflow_val_dataset = mlflow.data.from_pandas(X_val.join(y_val), source="validation", name="validation_dataset")
mlflow_test_dataset = mlflow.data.from_pandas(X_final.join(y_final), source="final_test", name="final_test_dataset")

# Değerlendirme metrikleri
def evaluate_model(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }

# Model tanımları
model_defs = {
    "LinearRegression": (LinearRegression, {}),
    
    "DecisionTree": (DecisionTreeRegressor, {
        'max_depth': hp.choice('dt_max_depth', [10, 20, 30]),
        'min_samples_split': hp.quniform('dt_min_samples_split', 2, 5, 1),
        'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 1, 4, 1)
    }),
    
    "RandomForest": (RandomForestRegressor, {
        'n_estimators': hp.choice('rf_n_estimators', [100, 200]),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 5, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 4, 1)
    }),
    
    "XGBoost": (xgb.XGBRegressor, {
        'max_depth': hp.choice('xgb_max_depth', [3, 5, 7]),
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'subsample': hp.uniform('xgb_subsample', 0.5, 1)
    })
}


model_run_count = {}

# Model çalıştırma
for name, (model_class, space) in model_defs.items():
    model_run_count[name] = model_run_count.get(name, 0) + 1
    run_idx = model_run_count[name]

    run_name_default = f"{name.lower()}_{run_idx}_default"
    model_name_default = f"{name.lower()}_{run_idx}_regressor"

    print(f"\n🚀 Model: {name} Run {run_idx}")

    with mlflow.start_run(run_name=run_name_default) as run:
        model = model_class()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        metrics = evaluate_model(y_val, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, "model", registered_model_name=model_name_default)
        mlflow.log_input(mlflow_train_dataset)
        mlflow.log_input(mlflow_val_dataset)
        print(f"📌 {run_name_default} → MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}")

    if space:
        def objective_with_name(params):
            model_run_count[name] += 1
            idx = model_run_count[name]
            run_name = f"{name.lower()}_{idx}_optimized"
            model_reg_name = f"{name.lower()}_{idx}_regressor"

            for key in ["min_samples_split", "min_samples_leaf", "n_estimators"]:
                if key in params:
                    params[key] = int(params[key])

            model = model_class(**params)
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                metrics = evaluate_model(y_val, y_pred)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, "model", registered_model_name=model_reg_name)
                mlflow.log_input(mlflow_train_dataset)
                mlflow.log_input(mlflow_val_dataset)
                print(f"🔍 {run_name} → MAE: {mae:.2f} | R²: {metrics['r2']:.3f}")

            return {'loss': mae, 'status': STATUS_OK}

        trials = Trials()
        fmin(
            fn=objective_with_name,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )

# 🔍 MLflow üzerinden en iyi modeli seç



runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.mae ASC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"\n🌟 Best Run ID: {best_run_id} | MAE: {best_run.data.metrics['mae']:.2f}")

# Modeli yükle
best_model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(best_model_uri)

# Final test verisiyle değerlendirme
y_pred_final = loaded_model.predict(X_final)
final_metrics = evaluate_model(y_final, y_pred_final)

with mlflow.start_run(run_name="Final_Test_Evaluation") as run:
    mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
    mlflow.sklearn.log_model(loaded_model, "final_model", registered_model_name="Best_Final_Model")
    mlflow.log_input(mlflow_test_dataset)
    print(f"\n🏁 Final Test Set → MAE: {final_metrics['mae']:.2f} | R²: {final_metrics['r2']:.3f}")

print("\n✅ Workflow completed and best model evaluated on final test set.")