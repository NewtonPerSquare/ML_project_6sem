#Паскаль Егор
import argparse
import yaml
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки конфига: {str(e)}")
        sys.exit(1)


def run_inference(file_path):
    """Режим прогнозирования"""
    logging.info(f"Запуск прогнозирования для файла: {file_path}")
    try:
        result = subprocess.run(
            [
                'python', 'model_serving.py',
                '-mode', 'inference',
                '-file', file_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка прогнозирования: {e.stderr}")
        sys.exit(1)


def run_update():
    """Режим обновления модели"""
    logging.info("Запуск обновления модели")
    try:
        #обработка данных
        subprocess.run(['python', 'data_collection.py'], check=True)
        subprocess.run(['python', 'data_analysis.py'], check=True)
        subprocess.run(['python', 'data_preprocessing.py'], check=True)

        #обучение модели
        result = subprocess.run(
            ['python', 'model_training.py'],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(result.stdout)

        #валидация
        subprocess.run(['python', 'model_validation.py'], check=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка обновления: {e.stderr}")
        sys.exit(1)


def run_summary():
    """Формирование сводного отчета"""
    logging.info("Генерация отчетов")
    try:
        config = load_config()

        #пути из конфига
        data_quality_path = Path(config['data_analysis']['report_dir']) / 'data_quality_report.json'
        model_metrics_path = Path(config['model_training']['metrics_path'])
        validation_report_path = Path(config['model_validation']['validation_report_path'])

        #проверка существования файлов
        if not data_quality_path.exists():
            raise FileNotFoundError(f"Отчет о качестве данных не найден: {data_quality_path}")

        # Сбор всех отчетов
        reports = {
            'data_quality': data_quality_path.read_text(encoding='utf-8'),
            'model_metrics': model_metrics_path.read_text(encoding='utf-8'),
            'validation': validation_report_path.read_text(encoding='utf-8')
        }

        # Сохранение объединенного отчета
        summary_path = Path('reports') / 'summary_report.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Consolidated Report\n\n")
            f.write("## Data Quality\n```json\n")
            f.write(reports['data_quality'])
            f.write("\n```\n\n")
            f.write("## Model Metrics\n```json\n")
            f.write(reports['model_metrics'])
            f.write("\n```\n\n")
            f.write("## Validation Results\n```json\n")
            f.write(reports['validation'])
            f.write("\n```\n")

        logging.info(f"Отчет сохранен в {summary_path}")

    except Exception as e:
        logging.error(f"Ошибка генерации отчета: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline Manager")
    parser.add_argument('-mode', required=True,
                        choices=['inference', 'update', 'summary'],
                        help="Режим работы: inference/update/summary")
    parser.add_argument('-file', help="Путь к данным для прогноза (требуется для inference)")

    args = parser.parse_args()
    config = load_config()

    if args.mode == 'inference':
        if not args.file:
            logging.error("Для режима inference требуется параметр -file")
            sys.exit(1)
        run_inference(args.file)
    elif args.mode == 'update':
        run_update()
    elif args.mode == 'summary':
        run_summary()