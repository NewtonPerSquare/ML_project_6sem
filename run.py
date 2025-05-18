# Паскаль Егор
import argparse
import logging
import subprocess
import sys
from pathlib import Path
import glob
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


MODEL_TYPES = ['xgboost', 'linear', 'decision_tree'] # Модели машинного обучения


def load_config():
    """Загружает конфигурацию из YAML"""
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Config load error: {str(e)}")
        sys.exit(1)


def run_inference(file_path, model_type):
    """Режим прогнозирования"""
    logging.info(f"Starting prediction for {model_type} model")
    try:
        result = subprocess.run(
            [
                sys.executable, 'model_serving.py',
                '-mode', 'inference',
                '-file', file_path,
                '-model', model_type
            ],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logging.info(f"Model output:\n{result.stdout}")
            print(f"\nModel output:\n{result.stdout}")  # Вывод в консоль

        if result.stderr:
            logging.warning(f"Model warnings:\n{result.stderr}")
            print(f"Warnings:\n{result.stderr}")  # Вывод в консоль

        return True

    except subprocess.CalledProcessError as e:
        # Обработка ошибок выполнения
        error_msg = f"Inference failed: {e.stderr}" if e.stderr else "Unknown error"
        logging.error(error_msg)
        print(f"\nERROR: {error_msg}")
        return False


def run_update():
    """Режим обновления модели"""
    logging.info("Starting full pipeline update")
    success = True

    try:
        # Обработка данных
        commands = [
            ([sys.executable, 'data_collection.py'], "Data collection"),
            ([sys.executable, 'data_analysis.py'], "Data analysis"),
            ([sys.executable, 'data_preprocessing.py'], "Data preprocessing")
        ]

        for cmd, name in commands:
            logging.info(f"Running {name}...")
            print(f"\n=== {name} ===")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            # Обработка вывода
            _handle_subprocess_output(result, name)
            if result.returncode != 0:
                success = False
                break

        # Обучение моделей
        if success:
            for model_type in MODEL_TYPES:
                logging.info(f"Training {model_type} model...")
                print(f"\n=== Training {model_type} ===")
                result = subprocess.run(
                    [sys.executable, 'model_training.py', '-model_type', model_type],
                    capture_output=True,
                    text=True
                )
                _handle_subprocess_output(result, f"Training {model_type}")
                if result.returncode != 0:
                    success = False
                    break

        # Валидация моделей
        if success:
            for model_type in MODEL_TYPES:
                logging.info(f"Validating {model_type} model...")
                print(f"\n=== Validating {model_type} ===")
                result = subprocess.run(
                    [sys.executable, 'model_validation.py', '-model_type', model_type],
                    capture_output=True,
                    text=True
                )
                _handle_subprocess_output(result, f"Validation {model_type}")
                if result.returncode != 0:
                    success = False
                    break

        return success

    except Exception as e:
        logging.error(f"Update error: {str(e)}")
        print(f"\nCRITICAL ERROR: {str(e)}")
        return False


def _handle_subprocess_output(result, process_name):
    """Обработка вывода подпроцессов"""
    output = []
    if result.stdout:
        output.append(f"{process_name} output:\n{result.stdout}")
    if result.stderr:
        output.append(f"{process_name} errors:\n{result.stderr}")

    full_output = "\n".join(output)

    if result.returncode == 0:
        logging.info(full_output)
        print(full_output)  # Выводим в консоль
    else:
        logging.error(full_output)
        print(full_output)  # Выводим в консоль


def run_summary():
    """Формирование сводного отчета"""
    logging.info("Generating summary report")
    try:
        config = load_config()
        report_sections = []

        # Отчет о качестве данных
        data_quality_path = Path(config['data_analysis']['report_dir']) / 'data_quality_report.json'
        if data_quality_path.exists():
            report_sections.append(
                "## Data Quality Report\n```json\n" +
                data_quality_path.read_text(encoding='utf-8') +
                "\n```"
            )

        # Метрики моделей
        model_metrics = {}
        for model_type in MODEL_TYPES:
            metric_path = Path(f"reports/model_metrics_{model_type}.json")
            if metric_path.exists():
                model_metrics[model_type] = metric_path.read_text(encoding='utf-8')

        # Отчеты валидации
        validation_reports = {}
        for report_path in glob.glob("reports/validation_report_*.json"):
            model_type = report_path.split("_")[-1].split(".")[0]
            validation_reports[model_type] = Path(report_path).read_text(encoding='utf-8')

        # Формирование отчета
        summary = ["# MLOps Pipeline Summary Report\n"]

        # Секция качества данных
        if report_sections:
            summary.append("\n".join(report_sections))

        # Секция метрик моделей
        summary.append("\n## Model Performance Metrics")
        for model_type, metrics in model_metrics.items():
            summary.append(f"\n### {model_type.capitalize()} Model\n```json\n{metrics}\n```")

        # Секция валидации
        summary.append("\n## Validation Results")
        for model_type, report in validation_reports.items():
            summary.append(f"\n### {model_type.capitalize()} Validation\n```json\n{report}\n```")

        # Сохранение отчета
        summary_path = Path('reports') / 'summary_report.md'
        summary_path.write_text("\n".join(summary), encoding='utf-8')

        logging.info(f"Summary report saved to {summary_path}")
        return True

    except Exception as e:
        logging.error(f"Summary generation error: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline Manager")
    parser.add_argument('-mode', required=True,
                        choices=['inference', 'update', 'summary'],
                        help="Режим работы: inference/update/summary")
    parser.add_argument('-file', help="Путь к данным для прогноза (требуется для inference)")
    parser.add_argument('-model',
                        choices=MODEL_TYPES,
                        default='xgboost',
                        help="Выбор модели для inference (по умолчанию: xgboost)")
    parser.add_argument('-iterations', type=int, default=1, help='Number of training iterations')
    args = parser.parse_args()
    config = load_config()

    if args.mode == 'inference':
        if not args.file:
            logging.error("Для режима inference необходимо указать -file")
            sys.exit(1)
        success = run_inference(args.file, args.model)
    elif args.mode == 'update':
        success = run_update()
    elif args.mode == 'summary':
        success = run_summary()

    sys.exit(0 if success else 1)
