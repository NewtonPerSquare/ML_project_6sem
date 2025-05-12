# Паскаль Егор
import argparse
import joblib
import pandas as pd
import logging
import yaml
import os
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
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

class ModelService:
    """Класс обслуживания модели"""
    def __init__(self, config, model_type):
        self.config = config['model_serving']
        self.model = joblib.load(self.config['model_paths'][model_type])
        self.preprocessor = joblib.load(self.config['preprocessor_path'])
        logging.info("Model loaded successfully")

    def predict(self, data_path, config, model_type):
        """Предсказания"""
        try:
            # Определение кодировки и чтение данных
            with open(data_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            logging.info(f"Detected encoding: {encoding}")

            df = pd.read_csv(
                data_path,
                encoding=encoding,
                engine='python',
                on_bad_lines='warn'
            )
            logging.info(f"Initial rows: {len(df)}")
            if df.empty:
                raise ValueError("Input file is empty")

            # Проверка обязательных колонок
            required_cols = ['Order Date', 'Ship Date']
            if not all(col in df.columns for col in required_cols):
                missing = list(set(required_cols) - set(df.columns))
                raise ValueError(f"Missing columns: {missing}")

            # Обработка дат
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
            if invalid_dates.any().any():
                logging.warning(f"Некорректные даты в строках: {invalid_dates[invalid_dates].index.tolist()}")
                df = df.dropna(subset=['Order Date', 'Ship Date'])

                # Расчет времени доставки
            df['DeliveryTime'] = (df['Ship Date'] - df['Order Date']).dt.days
            df = self._add_features(df, config)
            logging.info("Features added successfully")

            df = df.dropna(subset=['DeliveryTime'])
            df = df[df['DeliveryTime'] >= 0]
            logging.info(f"Final rows: {len(df)}")
            if df.empty:
                raise ValueError("No valid delivery time data")

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

    def _add_features(self, df, config):
        """Генерация новых признаков"""
        df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek

        bins = config['data_analysis'].get('delivery_bins', [0, 3, 7, 100])
        labels = ["Fast", "Medium", "Slow"]
        df["DeliverySpeed"] = pd.cut(df['DeliveryTime'], bins=bins, labels=labels)
        return df


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
        predictions, df = service.predict(args.file, config, args.model)

        output_path = f"predictions/predictions_{args.model}_{os.path.basename(args.file)}"
        df['PredictedDeliveryTime'] = predictions
        pd.DataFrame(predictions, columns=['PredictedDeliveryTime']).to_csv(output_path, index=False)

    except Exception as e:
        print(f"Err: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
