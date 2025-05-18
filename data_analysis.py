# Паскаль Егор
import pandas as pd
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Настройка логирования
logging.basicConfig(
    filename='logs/data_analysis.log',
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


def get_previous_batch(current_path):
    """Находит предыдущий батч по имени файлов"""
    batch_dir = os.path.dirname(current_path)
    batches = sorted([f for f in os.listdir(batch_dir) if f.startswith('batch_')])
    try:
        current_idx = batches.index(os.path.basename(current_path))
        return os.path.join(batch_dir, batches[current_idx - 1]) if current_idx > 0 else None
    except ValueError:
        return None


def add_features(df, config):
    """Создание новых признаков"""
    # День недели заказа
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek  # 0 = Пн, 6 = Вс

    # Время доставки категориальное
    bins = config['data_analysis'].get('delivery_bins', [0, 3, 7, float("inf")])
    labels = ["Fast", "Medium", "Slow"]
    df["DeliverySpeed"] = pd.cut(df["DeliveryTime"], bins=bins, labels=labels)
    return df


def generate_eda_report(df, batch_id, report_dir):
    """Генерация EDA-отчета с графиками"""
    eda_dir = os.path.join(report_dir, "eda", batch_id)
    os.makedirs(eda_dir, exist_ok=True)

    if "DeliveryTime" not in df.columns:
        logging.error("DeliveryTime column missing!")
        return

    # Распределение DeliveryTime
    plt.figure(figsize=(10, 6))
    sns.histplot(df["DeliveryTime"], kde=True)
    plt.title("Delivery Time Distribution")
    plt.savefig(os.path.join(eda_dir, "delivery_time_dist.png"))
    plt.close()

    # Корреляционная матрица
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(eda_dir, "correlation_matrix.png"))
        plt.close()

    # Категориальные признаки
    categorical_stats = {}
    for col in df.select_dtypes(include="object").columns:
        categorical_stats[col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }

    with open(os.path.join(eda_dir, "categorical_stats.json"), "w") as f:
        json.dump(categorical_stats, f)


def detect_data_drift(current_df, previous_batch_path):
    """Обнаружение дрифта данных"""
    if not previous_batch_path or not os.path.exists(previous_batch_path):
        return None

    try:
        prev_df = pd.read_csv(previous_batch_path)
        drift_report = {}

        # Для числовых признаков
        for col in ["DeliveryTime", "Sales"]:
            if col in current_df.columns and col in prev_df.columns:
                drift_report[col] = {
                    "wasserstein": wasserstein_distance(
                        current_df[col].dropna(),
                        prev_df[col].dropna()
                    )
                }

        # Для категориальных
        for col in ["Ship Mode", "Category"]:
            if col in current_df.columns and col in prev_df.columns:
                # Объединяем уникальные категории из обоих батчей
                all_categories = set(current_df[col]).union(set(prev_df[col]))

                # Создаем распределения с одинаковыми категориями
                current_counts = current_df[col].value_counts(normalize=True).reindex(all_categories, fill_value=0)
                prev_counts = prev_df[col].value_counts(normalize=True).reindex(all_categories, fill_value=0)

                # Проверка на пустые распределения
                if current_counts.sum() == 0 or prev_counts.sum() == 0:
                    continue

                drift_report[col] = {
                    "js_divergence": jensenshannon(current_counts, prev_counts)
                }

        return drift_report
    except Exception as e:
        logging.error(f"Drift detection error: {str(e)}")
        return None


def data_clean(df, batch_path, config):
    try:
        df = df.copy()
        required_columns = ['Order Date', 'Ship Date']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Missing columns {missing} in {os.path.basename(batch_path)}")
            return None

        date_format = config['data_analysis'].get('date_format', '%d-%m-%Y')
        df['Order Date'] = pd.to_datetime(df['Order Date'], format=date_format, errors='coerce')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format=date_format, errors='coerce')

        invalid_dates = df[['Order Date', 'Ship Date']].isnull().any(axis=1)
        if invalid_dates.any():
            df = df.dropna(subset=['Order Date', 'Ship Date'])

        df['DeliveryTime'] = (df['Ship Date'] - df['Order Date']).dt.days
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
        df = add_features(df, config)
        return df, invalid_dates

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(batch_path)}: {str(e)}")
        return None


def analyze_batch(batch_path, config):
    """Анализ одного батча, формирование отчета"""
    try:
        df = pd.read_csv(batch_path)
        initial_count = len(df)
        df, invalid_dates = data_clean(df, batch_path, config)
        os.makedirs('data/cleaned', exist_ok=True)
        clean_path = os.path.join('data/cleaned', os.path.basename(batch_path))
        df.to_csv(clean_path, index=False)

        report = {
            'initial_rows': initial_count,
            'final_rows': len(df),
            'removed_invalid_dates': int(invalid_dates.sum()),
            'removed_outliers': int((len(df) + invalid_dates.sum()) - initial_count)
        }

        previous_batch = get_previous_batch(batch_path)
        drift_report = detect_data_drift(df, previous_batch)
        report['drift_metrics'] = drift_report if drift_report else "No previous data"

        if config["data_analysis"].get("enable_eda", False):
            batch_id = os.path.basename(batch_path).split(".")[0]
            generate_eda_report(df, batch_id, config["data_analysis"]["report_dir"])

        return report

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(batch_path)}: {str(e)}")
        return None


if __name__ == "__main__":
    # Вспоминаем конфиг
    with open('config.yaml', 'r') as f:
        config = load_config()

    # И все, что с конфигом связано
    raw_data_dir = config['data_collection']['raw_data_dir']
    report_dir = config['data_analysis']['report_dir']

    # Ну, а вдруг директории нет...
    os.makedirs(report_dir, exist_ok=True)

    reports = {}
    for batch_file in os.listdir(raw_data_dir):
        if batch_file.endswith('.csv'):
            batch_path = os.path.join(raw_data_dir, batch_file)
            report = analyze_batch(batch_path, config)
            if report:
                reports[batch_file] = report

    # Сохраняем отчет
    report_path = os.path.join(report_dir, 'data_quality_report.json')
    report_path = report_path.replace('\\', '/') # Чтобы все слеши были в одну сторону
    with open(report_path, 'w') as f:
        json.dump(reports, f)
    logging.info(f"Report saved in {report_path}")
