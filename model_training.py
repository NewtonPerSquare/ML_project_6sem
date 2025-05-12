# Холевенкова Варвара
import pandas as pd
import os
import yaml
import logging
import joblib
import json
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import argparse

# Настройка логирования
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'a'
)

# Модели машинного обучения
MODELS = {
    'xgboost': xgb.XGBRegressor,
    'linear': LinearRegression,
    'decision_tree': DecisionTreeRegressor
}


def load_config():
    """Загружает конфигурацию из YAML"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_all_processed_data(processed_dir):
    """Загружает все обработанные батчи"""
    logging.info(f"Loading processed data from {processed_dir}")
    dfs = []
    for file in os.listdir(processed_dir):
        if file.startswith('processed_'):
            file_path = os.path.join(processed_dir, file)
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logging.debug(f"Loaded {file} with {len(df)} rows")
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
    return pd.concat(dfs, ignore_index=True)


def calculate_metrics(y_true, y_pred):
    """Вычисление метрик"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def save_metrics(metrics, path):
    """Добавляем логи сохранения метрик"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {str(e)}")
        raise

def train_model(full_config):
    """Обучение"""
    try:
        model_cfg = full_config['model_training']
        model_type = model_cfg['model_type']

        # Загрузка и разделение данных
        df = load_all_processed_data(model_cfg['processed_data_dir'])
        X = df.drop(model_cfg['target_column'], axis=1) # Работаем с предобработанными данными
        y = df[model_cfg['target_column']]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=model_cfg['test_size'],
            random_state=42
        )
        logging.info(f"Data split: train={len(X_train)}, val={len(X_val)}")
        logging.info(f"Starting training for model type: {model_type}")

        # Инициализация
        model_path = f"models/{model_type}.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info(f"Loaded existing {model_type} model for update")

            # Обновление модели
            if model_type == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    xgb_model=model.get_booster(),
                    **model_cfg['training_params']
                )
                logging.info("Updated XGBoost model with new trees")
            else:
                model = clone(model).fit(X_train, y_train)
                logging.info(f"Retrained {model_type} model from scratch")
        else:
            model_class = MODELS[model_type]
            model = model_class(**model_cfg['model_params'][model_type])
            if model_type == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    **model_cfg['training_params']
                )
            else:
                model.fit(X_train, y_train)
            logging.info(f"Initialized new {model_type} model")

        # Валидация и сохранение
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)

        logging.info("Validation metrics:\n" +
                     "\n".join([f"{k}: {v}" for k, v in metrics.items()]))

        save_metrics(metrics, model_path)
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-model_type',
                        required=True,
                        choices=['xgboost', 'linear', 'decision_tree'],
                        help='Тип модели для обучения')
    args = parser.parse_args()

    try:
        config = load_config()
        config['model_training']['model_type'] = args.model_type  # Переопределяем тип из конфига
        train_model(config)
    except Exception as e:
        logging.error(f"Err: {str(e)}")
