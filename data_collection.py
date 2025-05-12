# Холевенкова Варвара
import chardet
import pandas as pd
import os
import json
import yaml
import logging

# Настройка логирования
logging.basicConfig(
    filename='logs/data_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'a'
)


def load_config():
    """Загружает конфигурацию из YAML"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def prepare_directories(config):
    """Создает директории для данных"""
    os.makedirs(config['raw_data_dir'], exist_ok=True)
    os.makedirs(config['metadata_dir'], exist_ok=True)


def detect_encoding(file_path):
    """Определяем кодировку (без этого постоянно падает)"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def process_data(input_path, raw_data_dir, metadata_dir, config):
    """Минимальная обработка и формирование батчей"""
    encoding = detect_encoding(input_path)

    # Читаем CSV с учётом кодировки
    df = pd.read_csv(
        input_path,
        encoding=encoding,
        engine='python',
        on_bad_lines='warn'
    )

    # Преобразование дат и вычисление времени доставки
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
    df['DeliveryTime'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Удаление строк с некорректными датами
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    allowed_windows = ['D', 'W', 'M'] # day, week, month
    if config["batch_window"] not in allowed_windows:
        raise ValueError(f"Invalid batch_window: {config['batch_window']}")

    # Разбивка на батчи по окну батча из config'a
    df["batch_window"] = df["Order Date"].dt.to_period(config["batch_window"])
    batches = df.groupby("batch_window")

    for period, batch in batches:
        safe_period = str(period).replace("-", "_").replace("/", "_")
        batch_file = os.path.join(raw_data_dir, f'batch_{safe_period}.csv')
        batch.drop('batch_window', axis=1).to_csv(batch_file, index=False)
        # Метаданные (ого)
        metadata_file = os.path.join(metadata_dir, f'metadata_{safe_period}.json')
        metadata = {
            'batch_id': str(period),
            'num_records': len(batch),
            'delivery_time_stats': {
                'mean': float(batch['DeliveryTime'].mean()),
                'max': int(batch['DeliveryTime'].max()),
                'min': int(batch['DeliveryTime'].min())
            }
        }

        # Обработка NaN (мало ли)
        for k, v in metadata['delivery_time_stats'].items():
            if pd.isna(v):
                metadata['delivery_time_stats'][k] = None

        # Запись в JSON
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        logging.info(f"Saved batch {period} with {len(batch)} records.")


if __name__ == "__main__":
    config = load_config()['data_collection']
    prepare_directories(config)
    try:
        process_data(
            input_path=config['input_data_path'],
            raw_data_dir=config['raw_data_dir'],
            metadata_dir=config['metadata_dir'],
            config=config
        )
        logging.info("Data collection ended successfully")
    except Exception as e:
        logging.error(f"Err: {str(e)}")
