# Холевенкова Варвара
import json
import joblib
import pandas as pd
import logging
import yaml
from datetime import datetime
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import shap
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(
    filename='logs/model_validation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'a'
)


MODEL_TYPES = ['xgboost', 'linear', 'decision_tree'] # Модели машинного обучения


def load_config():
    """Загружает конфигурацию из YAML"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true, y_pred):
    """Вычисление метрик"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def validate_model(config, model_type):
    """Валидация модели с автоматическим определением типа данных"""
    try:
        # Загрузка модели и предобработчика
        model_cfg = config['model_validation']

        # Путь к модели
        model_path = f"models/{model_type}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        # Загрузка модели
        model = joblib.load(model_path)

        # Загрузка предобработчика
        preprocessor = joblib.load(model_cfg['preprocessor_path'])

        # Загрузка данных
        if os.path.exists(model_cfg['validation_data_path']):
            df = pd.read_csv(model_cfg['validation_data_path'])
        else:
            processed_dir = config['model_training']['processed_data_dir']
            files = [f for f in os.listdir(processed_dir) if f.startswith('processed_')]
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_dir, x)))
            df = pd.read_csv(os.path.join(processed_dir, latest_file))

            # Сохраняем валидационную выборку
            os.makedirs(os.path.dirname(model_cfg['validation_data_path']), exist_ok=True)
            df.to_csv(model_cfg['validation_data_path'], index=False)
            logging.info(f"Saved validation data to {model_cfg['validation_data_path']}")

        # Применение предобработки
        X = df.drop(model_cfg['target_column'], axis=1)
        y = df[model_cfg['target_column']].values

        # Прогнозирование
        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)

        # Формирование отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_version': model_cfg['model_version'],
            'metrics': metrics,
            'features': preprocessor.get_feature_names_out().tolist(),
            'training_params': config['model_training']['model_params'].get(model_type, {})
        }

        # SHAP-анализ для tree-моделей
        if model_type in ['xgboost', 'decision_tree']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=preprocessor.get_feature_names_out(), show=False)
            shap_path = f"reports/shap_{model_type}.png"
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            report['shap_analysis'] = shap_path

        # Коэффициенты для линейной регрессии
        elif model_type == 'linear' and hasattr(model, 'coef_'):
            report['coefficients'] = dict(zip(
                preprocessor.get_feature_names_out(),
                model.coef_.round(4).tolist()
            ))

        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.3)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plot_path = f"reports/validation_plot_{model_type}.png"
        plt.savefig(plot_path)
        plt.close()
        report['validation_plot'] = plot_path

        # Сохранение отчета
        report_path = model_cfg['validation_report_path'].replace(".json", f"_{model_type}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logging.info(f"Validation report for {model_type} saved to {report_path}")
        return True

    except Exception as e:
        logging.error(f"Validation error for {model_type}: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Validation')
    parser.add_argument('-model_type',
                        required=True,
                        choices=MODEL_TYPES,
                        help='Тип модели для валидации: xgboost, linear, decision_tree')
    args = parser.parse_args()

    try:
        config = load_config()
        success = validate_model(config, args.model_type)
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
