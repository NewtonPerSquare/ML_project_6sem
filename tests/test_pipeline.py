import unittest
import os
import joblib

class TestPipeline(unittest.TestCase):
    def test_models_exist(self):
        """Проверка наличия и загружаемости моделей"""
        models = ['xgboost', 'linear', 'decision_tree']
        for model in models:
            path = f'models/{model}.joblib'
            self.assertTrue(os.path.exists(path), f"Модель {model} не найдена")
            try:
                joblib.load(path)
            except Exception as e:
                self.fail(f"Ошибка загрузки модели {model}: {str(e)}")

    def test_data_processed(self):
        """Проверка наличия обработанных данных"""
        self.assertTrue(os.path.exists('data/processed'), "Директория data/processed отсутствует")
        self.assertGreater(len(os.listdir('data/processed')), 0, "Нет обработанных файлов")

if __name__ == '__main__':
    unittest.main()
