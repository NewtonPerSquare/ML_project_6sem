#Холевенкова Варвара
import pandas as pd
import os
import yaml
import logging
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_all_processed_data(processed_dir):         #Загружает все обработанные батчи
    dfs = []
    for file in os.listdir(processed_dir):
        if file.startswith('processed_'):
            df = pd.read_csv(os.path.join(processed_dir, file))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def train_model(full_config):           #Обучение
    model_cfg = full_config['model_training']
    prep_cfg = full_config['data_preprocessing']

    #Загрузка данных
    df = load_all_processed_data(model_cfg['processed_data_dir'])
    X = df.drop(model_cfg['target_column'], axis=1)
    y = df[model_cfg['target_column']]

    #Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=model_cfg['test_size'],
        random_state=42
    )

    #Инициализация
    if os.path.exists(model_cfg['model_path']):
        model = joblib.load(model_cfg['model_path'])
        logging.info("Дообучение модели...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **model_cfg['training_params']
        )
    else:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            enable_categorical=True,
            **model_cfg['model_params']
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **model_cfg['training_params']
        )
        logging.info("Инициализирована новая модель XGBoost")

    #Валидация
    y_pred = model.predict(X_val)
    metrics = {
        'MAE': mean_absolute_error(y_val, y_pred),
        'MSE': mean_squared_error(y_val, y_pred),
    }

    #Сохранение
    model.save_model(model_cfg['model_path'] + ".ubj")
    with open(model_cfg['metrics_path'], 'w') as f:
        json.dump(metrics, f)

    logging.info(f"Метрики: {metrics}")


if __name__ == "__main__":
    try:
        full_config = load_config()
        train_model(full_config)
        logging.info("Обучение успешно завершено")
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
