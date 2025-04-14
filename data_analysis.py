#Паскаль Егор
import pandas as pd
import json
import os
import yaml
import logging

#настройка логирования
logging.basicConfig(
    filename='logs/data_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config():
    """Выгрузка конфига"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def analyze_batch(batch_path):
    """Анализ одного батча, формирование отчета"""
    df = pd.read_csv(batch_path)
    report = {
        'missing_values': df.isnull().sum().to_dict(),      #подсчёт пропусков (неожиданно)
        'outliers': {}
    }

    #используется iqr для поиска выбросов
    q1 = df['DeliveryTime'].quantile(0.25)
    q3 = df['DeliveryTime'].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr
    outliers = df[(df['DeliveryTime'] < low) | (df['DeliveryTime'] > up)]       #соответственно, выбросы
    report['outliers']['DeliveryTime'] = len(outliers)                          #всё записываем (а то забудем)
    return report


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:                                         #вспоминаем конфиг
        config = yaml.safe_load(f)

    raw_data_dir = config['data_collection']['raw_data_dir']                    #и все, что с конфигом связано
    report_dir = config['data_analysis']['report_dir']

    os.makedirs(report_dir, exist_ok=True)                                      #ну, а вдруг директории нет...

    all_reports = {}
    batch_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]

    for batch_file in batch_files:
        batch_path = os.path.join(raw_data_dir, batch_file)
        try:
            report = analyze_batch(batch_path)
            all_reports[batch_file] = report
            logging.info(f"Проанализирован {batch_file}")
        except Exception as e:
            logging.error(f"Ошибка в {batch_file}: {str(e)}")

    report_path = os.path.join(report_dir, 'data_quality_report.json')          #сохраняем отчет
    with open(report_path, 'w') as f:
        json.dump(all_reports, f)
    logging.info(f"Отчет сохранен в {report_path}")
