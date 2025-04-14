#Паскаль Егор
import argparse
import joblib
import pandas as pd
import logging
import yaml
import os
import chardet
import xgboost as xgb
import sklearn, xgboost

logging.basicConfig(
    filename='logs/model_serving.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ModelService:
    def __init__(self, config):
        self.model = xgb.XGBRegressor()
        self.model.load_model(config['model_path'] + ".ubj")
        self.preprocessor = joblib.load(config['preprocessor_path'])

    def predict(self, data_path):
        #определяем кодировку файла
        with open(data_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']

        #читаем файл с определенной кодировкой (иначе боль)
        df = pd.read_csv(
            data_path,
            encoding=encoding,
            engine='python',
            on_bad_lines='warn'
        )

        df['DeliveryTime'] = (
                pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
                - pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
        ).dt.days
        df = df.dropna(subset=['DeliveryTime'])

        X = self.preprocessor.transform(df)
        return self.model.predict(X), df


def main():
    parser = argparse.ArgumentParser(description='ML Model Serving')
    parser.add_argument('-mode', required=True, choices=['inference'], help='Режим работы')
    parser.add_argument('-file', required=True, help='Путь к входному файлу')
    args = parser.parse_args()

    config = yaml.safe_load(open('config.yaml'))['model_serving']
    service = ModelService(config)

    try:
        predictions, df = service.predict(args.file)
        output_path = f"predictions/predictions_{os.path.basename(args.file)}"
        pd.DataFrame(predictions, columns=['PredictedDeliveryTime']).to_csv(output_path, index=False)
        print(f"Прогнозы сохранены в: {output_path}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()