#Холевенкова Варвара
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
                                                                            #Настройка логирования
logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def detect_encoding(file_path):                             #Определяем кодировку (без этого постоянно падает)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def create_preprocessor(config):                                            #Пайплайн предобработки(простенькой) данных
    numeric_features = config['numeric_features']
    categorical_features = config['categorical_features']

    numeric_transformer = Pipeline(steps=[                                  #Для целочисленных признаков
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[                              #Для категориальных
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def process_batches(config):                                        #Побатчевая предобработка
    raw_data_dir = config['data_collection']['raw_data_dir']
    processed_dir = config['data_preprocessing']['processed_dir']
    target_column = config['model_training']['target_column']

    os.makedirs(processed_dir, exist_ok=True)

    #Сбор и проверка данных
    all_data = []
    for batch_file in os.listdir(raw_data_dir):
        if batch_file.endswith('.csv'):
            batch_path = os.path.join(raw_data_dir, batch_file)
            df = pd.read_csv(batch_path)

            #Проверка структуры данных
            required_cols = (
                    config['data_preprocessing']['numeric_features'] +
                    config['data_preprocessing']['categorical_features'] +
                    [target_column]
            )
            if not set(required_cols).issubset(df.columns):
                missing = set(required_cols) - set(df.columns)
                logging.error(f"Missing columns in {batch_file}: {missing}")
                continue

            all_data.append(df)

    #Обучение предобработчика
    full_df = pd.concat(all_data)
    preprocessor = create_preprocessor(config['data_preprocessing'])
    preprocessor.fit(full_df.drop(columns=[target_column]))
    feature_names = get_feature_names(preprocessor)

    #Сохранение эталонной размерности (проверка на адекватность (думаю, можно и убрать...))
    expected_shape = (len(full_df), len(feature_names))
    logging.info(f"Expected shape after transformation: {expected_shape}")

    #Обработка батчей с валидацией
    for batch_file in os.listdir(raw_data_dir):
        if not batch_file.endswith('.csv'):
            continue

        batch_path = os.path.join(raw_data_dir, batch_file)
        try:
            encoding = detect_encoding(batch_path)
            df = pd.read_csv(batch_path,
                encoding=encoding,
                engine='python',
                on_bad_lines='skip',
                dtype_backend='pyarrow',
                skip_blank_lines=True
            )

            X = df.drop(columns=[target_column])
            y = df[target_column]

            #Преобразование и проверка
            X_transformed = preprocessor.transform(X)

            #Критическая проверка размерности
            if X_transformed.shape[1] != len(feature_names):
                error_msg = f"""
                Dimension mismatch! 
                Expected: {len(feature_names)}, Actual: {X_transformed.shape[1]}
                Batch: {batch_file}
                Columns: {X.columns.tolist()}
                """
                logging.error(error_msg)
                continue

            #Создание DataFrame
            processed_df = pd.DataFrame(
                X_transformed,
                columns=feature_names
            )
            processed_df[target_column] = y.values

            #Сохранение
            output_path = os.path.join(processed_dir, f'processed_{batch_file}')
            processed_df.to_csv(output_path, index=False)
            logging.info(f"Success: {batch_file}")

        except Exception as e:
            logging.error(f"Failed {batch_file}: {str(e)}")
            continue

    #Сохранение предобработчика
    joblib.dump(preprocessor, config['data_preprocessing']['preprocessor_path'])



def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop':
            continue

        if isinstance(transformer, Pipeline):
            #Обрабатываем каждый шаг пайплайна
            current_features = list(features)
            for step in transformer.steps:
                if hasattr(step[1], 'get_feature_names_out'):
                    current_features = step[1].get_feature_names_out(current_features)
                elif hasattr(step[1], 'get_feature_names'):
                    current_features = step[1].get_feature_names(current_features)
                else:
                    #Для SimpleImputer и других трансформеров без изменения имен
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