# Холевенкова Варвара
import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import chardet

# Настройка логирования
logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'a'
)


def load_config():
    """Загружает конфигурацию из YAML"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def detect_encoding(file_path):
    """Определяем кодировку (без этого постоянно падает)"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def create_preprocessor(config):
    """Пайплайн предобработки(простенькой) данных"""
    numeric_features = config['numeric_features']
    categorical_features = config['categorical_features']

    # Для целочисленных признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['na_handling']['numeric'])),
        ('scaler', StandardScaler())
    ])

    # Для категориальных
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(
            strategy=config['na_handling']['categorical'],
            fill_value='Unknown'
        )),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def process_batches(config):
    """Побатчевая предобработка данных с обработкой пропусков"""
    try:
        # Получаем конфигурации из полного конфига
        preprocessing_config = config['data_preprocessing']
        model_config = config['model_training']

        raw_data_dir = 'data/cleaned'
        processed_dir = preprocessing_config['processed_dir']
        target_column = model_config['target_column']

        os.makedirs(processed_dir, exist_ok=True)


        # Сбор данных с валидацией
        all_data = []
        for batch_file in os.listdir(raw_data_dir):
            if not batch_file.startswith('batch_'):
                continue

            batch_path = os.path.join(raw_data_dir, batch_file)
            try:
                # Чтение с обработкой ошибок
                df = pd.read_csv(
                    batch_path,
                    engine='python',
                    on_bad_lines='skip',
                    dtype_backend='pyarrow'
                )

                # Проверка обязательных колонок
                required_cols = (
                        preprocessing_config['numeric_features'] +
                        preprocessing_config['categorical_features'] +
                        [target_column]
                )
                missing = set(required_cols) - set(df.columns)
                if missing:
                    logging.warning(f"Skipping {batch_file} - missing: {missing}")
                    continue

                # Заполнение NA для новых фичей
                df['Order_DayOfWeek'] = df['Order_DayOfWeek'].fillna(-1).astype(int)
                if 'DeliverySpeed' in df:
                    df['DeliverySpeed'] = df['DeliverySpeed'].fillna('Unknown')

                # Удаление строк с NA в таргете
                df = df.dropna(subset=[target_column])

                all_data.append(df)
                logging.info(f"Processed {batch_file}")

            except Exception as e:
                logging.error(f"Failed {batch_file}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No valid data batches found after preprocessing")

        # Обучение препроцессора на полных данных
        full_df = pd.concat(all_data)
        preprocessor = create_preprocessor(preprocessing_config)  # Передаем конфиг предобработки
        preprocessor.fit(full_df.drop(columns=[target_column]))

        # Сохранение препроцессора
        joblib.dump(preprocessor, preprocessing_config['preprocessor_path'])

        # Обработка батчей
        for batch_file in os.listdir(raw_data_dir):
            if not batch_file.startswith('batch_'):
                continue

            batch_path = os.path.join(raw_data_dir, batch_file)
            try:
                df = pd.read_csv(batch_path)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Преобразование
                X_transformed = preprocessor.transform(X)

                # Создание финального датасета
                processed_df = pd.DataFrame(
                    X_transformed,
                    columns=preprocessor.get_feature_names_out()
                )
                processed_df[target_column] = y.values

                # Сохранение
                output_path = os.path.join(processed_dir, f'processed_{batch_file}')
                processed_df.to_csv(output_path, index=False)

            except Exception as e:
                logging.error(f"Final processing failed {batch_file}: {str(e)}")

    except Exception as e:
        logging.error(f"Critical error in preprocessing: {str(e)}")
        raise


def get_feature_names(column_transformer):
    """Обработчик пайплайна"""
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop':
            continue

        if isinstance(transformer, Pipeline):
            # Обрабатываем каждый шаг пайплайна
            current_features = list(features)
            for step in transformer.steps:
                if hasattr(step[1], 'get_feature_names_out'):
                    current_features = step[1].get_feature_names_out(current_features)
                elif hasattr(step[1], 'get_feature_names'):
                    current_features = step[1].get_feature_names(current_features)
                else:
                    # Для SimpleImputer и других трансформеров без изменения имен
                    current_features = current_features
            feature_names.extend(current_features)
        else:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(features)
            elif hasattr(transformer, 'get_feature_names'):
                names = transformer.get_feature_names(features)
            else:
                names = features
            feature_names.extend(names)
    return feature_names


if __name__ == "__main__":
    try:
        config = load_config()
        process_batches(config)
        logging.info("Data preprocessing completed successfully")
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
