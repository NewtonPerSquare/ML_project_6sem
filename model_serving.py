# Паскаль Егор
import argparse
import joblib
import pandas as pd
import logging
import os
import yaml
import sys
import chardet

# Настройка логирования
logging.basicConfig(
    filename='logs/model_serving.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'a'
)


def load_config():
    """Загружает конфигурацию из YAML"""
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Config load error: {str(e)}")
        sys.exit(1)


def detect_encoding(file_path):
    """Определяем кодировку (без этого постоянно падает)"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def add_features(df, config):
    """Создание новых признаков"""
    # День недели заказа
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek  # 0 = Пн, 6 = Вс

    # Время доставки категориальное
    bins = config['data_analysis'].get('delivery_bins', [0, 3, 7, 100])
    labels = ["Fast", "Medium", "Slow"]
    df["DeliverySpeed"] = pd.cut(df["DeliveryTime"], bins=bins, labels=labels)
    return df


def data_clean(df, batch_path, config):
    try:
        df = df.copy()
        required_columns = ['Order Date', 'Ship Date']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Missing columns {missing} in {os.path.basename(batch_path)}")
            return None

        df['Order Date'] = pd.to_datetime(
            df['Order Date'],
            format='%d-%m-%Y',
            errors='coerce'
        )
        df['Ship Date'] = pd.to_datetime(
            df['Ship Date'],
            format='%d-%m-%Y',
            errors='coerce'
        )

        invalid_dates = df[['Order Date', 'Ship Date']].isnull().any(axis=1)
        if invalid_dates.any():
            df = df.dropna(subset=['Order Date', 'Ship Date'])

        df['DeliveryTime'] = (df['Ship Date'] - df['Order Date']).dt.days
        df = add_features(df, config)
        negative_mask = df['DeliveryTime'] < 0
        if negative_mask.any():
            df.loc[negative_mask, 'DeliveryTime'] = abs(df.loc[negative_mask, 'DeliveryTime'])

        q1 = df['DeliveryTime'].quantile(0.25)
        q3 = df['DeliveryTime'].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            logging.warning("IQR is zero, skipping outlier removal")
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df['DeliveryTime'] >= lower) & (df['DeliveryTime'] <= upper)]
        return df

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(batch_path)}: {str(e)}")
        return None


class ModelService:
    """Класс обслуживания модели"""
    def __init__(self, config, model_type):
        self.config = config['model_serving']
        self.model = joblib.load(self.config['model_paths'][model_type])
        self.preprocessor = joblib.load(self.config['preprocessor_path'])
        logging.info("Model loaded successfully")

    def predict(self, data_path, model_type, config):
        """Предсказания"""
        try:
            df = pd.read_csv(
                data_path,
                encoding=detect_encoding(data_path),
                engine='python',
                on_bad_lines='warn'
            )
            df = data_clean(df, data_path, config)

            # Препроцессинг и предсказание
            logging.info(f"Using model: {model_type}")
            X_transformed = self.preprocessor.transform(df)
            feature_names = self.preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X_transformed, columns=feature_names)

            predictions = self.model.predict(X_df) # Теперь передаем DataFrame с именами

            return predictions, df

        except Exception as e:
            logging.error(f"Data processing error: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='ML Model Serving')
    parser.add_argument('-mode', required=True, choices=['inference'], help='Режим работы')
    parser.add_argument('-file', required=True, help='Путь к входному файлу')
    parser.add_argument('-model',
                        choices=['xgboost', 'decision_tree', 'linear'],
                        default='xgboost',
                        help='Выбор модели: xgboost (по умолчанию), decision_tree, linear')
    args = parser.parse_args()

    try:
        config = load_config()
        service = ModelService(config, args.model)
        predictions, df = service.predict(args.file, args.model, config)

        output_path = f"predictions/predictions_{args.model}_{os.path.basename(args.file)}"
        df['PredictedDeliveryTime'] = predictions
        pd.DataFrame(predictions, columns=['PredictedDeliveryTime']).to_csv(output_path, index=False)

    except Exception as e:
        logging.error(f"Err: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
