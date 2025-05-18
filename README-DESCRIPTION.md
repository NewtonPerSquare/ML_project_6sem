В этом файле описана общая структура всего проекта


i.Структура датасета Global Super Store
Датасет содержит информацию о заказах интернет-магазина за период с 2011 по 2014 год.
Краткое описание основных колонок:

1. Информация о заказе
Row ID — Уникальный идентификатор строки в таблице (первичный ключ).

Order ID — Уникальный номер заказа. Один заказ может содержать несколько товаров.

Order Date — Дата оформления заказа.

Ship Date — Дата отправки заказа.

Ship Mode — Способ доставки

2. Информация о клиенте

Customer ID — Уникальный идентификатор клиента.

Customer Name — Имя клиента.

Segment — Сегмент клиента (например, "Consumer" (частные лица), "Corporate" (корпоративные)).

3. Географические данные

City — Город доставки.

State — Штат/регион доставки.

Country — Страна доставки.

Postal Code — Почтовый индекс (может отсутствовать).

Market — Региональный рынок (например, "APAC", "EU").

Region — Регион внутри рынка (например, "South", "West").

4. Информация о товаре

Product ID — Уникальный идентификатор товара.

Category — Категория товара (например, "Furniture", "Office Supplies").

Sub-Category — Подкатегория товара (например, "Chairs", "Paper").

Product Name — Название товара.

5. Финансовые-количественные показатели

Sales — Сумма продажи в денежном выражении.

Quantity — Количество товара в заказе.

Discount — Размер скидки на товар (в долях или процентах).

Profit — Прибыль от заказа.

Shipping Cost — Стоимость доставки.

ii. Целевая переменная и использумеые методы

Вводится колонка Delivery Time = Ship Date - Order Date. Она и является параметром, который будут предсказывать модели.

iii. Описание того, что сохраняется после каждого этапа

а) Data Collection

Сохраняются сырые батчи в data/raw_batches/batch_<период>.csv.

Помимо этого сохраняются метаданные батчей в data/metadata/metadata_<период>.json.

б) Data Analysis

Очищенные батчи в data/cleaned/batch_<период>.csv

Data quality отчёты в reports/data_quality/data_qualuty_report.json

EDA-отчёты в reports/eda/batch_<период>:

    -графики распределений delivery_time_dist.png

    -тепловую карту корреляций correlation_matrix.png

    -статистику категориальных признаков categorial_stats.json

в) Data Preprocessing

преобразованные данные в батчах в data/processed/processed_batch_<период>.csv

обученный предобработчик в models/preprocessor.joblib

г) Model Training

обученные модели в models/<модель>.joblib

метрики в reports/model_metrics_<модель>.json

д) Model Validation

отчет о валидации в reports/validation_report_<модель>.json

Визуализации:
    
    -SHAP-графики reports/shap_<модель>.png
    
    -Графики предсказаний reports/validation_plot_<модель>.png

е) Model Serving

предсказания predictions/predictions_<модель>_<имя исходного файла>.csv
