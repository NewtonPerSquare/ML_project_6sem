#Холевенкова Варвара
import json
import joblib
import pandas as pd
import logging
import yaml
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import os
from data_preprocessing import get_feature_names
import xgboost as xgb

logging.basicConfig(
    filename='logs/model_validation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def validate_model(config):     #Валидация модели с автоматическим определением типа данных
    try:
        #Загрузка модели и предобработчика
        model = xgb.XGBRegressor()
        model.load_model(config['model_path'] + ".json")
        preprocessor = joblib.load(config['preprocessor_path'])

        #Загрузка данных с автоматическим определением типа
        if os.path.exists(config['validation_data_path']):
            df = pd.read_csv(config['validation_data_path'])
            if 'DeliveryTime' not in df.columns:
                df['DeliveryTime'] = (pd.to_datetime(df['Ship Date'])
                                      - pd.to_datetime(df['Order Date'])).dt.days
            df = df.dropna(subset=['DeliveryTime'])
            X = preprocessor.transform(df.drop(config['target_column'], axis=1))
        else:
            logging.warning("Using processed training data for validation")
            processed_files = [f for f in os.listdir('data/processed')
                               if f.startswith('processed_')]
            latest_file = max(processed_files, key=lambda x: os.path.getmtime(
                os.path.join('data/processed', x)))
            df = pd.read_csv(os.path.join('data/processed', latest_file))
            X = df.drop(config['target_column'], axis=1).values  #Преобразованные данные

        y = df[config['target_column']].values

        #Прогноз
        logging.info(f"Прогноз для {len(df)} записей")
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)

        #Формирование отчета
        logging.info("Формирование отчета")
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_source': config['validation_data_path'] if os.path.exists(
                config['validation_data_path']) else latest_file,
            'model_version': config['model_version'],
            'MAE': round(mae, 4),
            'feature_importance': dict(zip(
                get_feature_names(preprocessor),
                model.feature_importances_.round(4).tolist()
            ))
        }

        #Сохранение результатов
        os.makedirs(os.path.dirname(config['validation_report_path']), exist_ok=True)
        with open(config['validation_report_path'], 'w') as f:
            json.dump(report, f, indent=2)

        logging.info(f"Validation successful. MAE: {mae:.2f}")
        return True

    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        config = load_config()['model_validation']
        validate_model(config)
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
