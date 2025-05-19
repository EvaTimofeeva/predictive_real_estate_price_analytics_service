"""This is full life cycle for ML model: parsing, preprocessing, training, serving."""

import os
import glob
import argparse
import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from logging.config import dictConfig
import matplotlib.pyplot as plt
from cianparser import CianParser
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Константы
TEST_SIZE = 0.2
MODEL_PATH = "models/linear_regression_model.pkl"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/progress/progress_data.csv"

# Инициализация парсера CIAN
moscow_parser = CianParser(location="Москва")


def is_folder_empty(folder_path: str) -> bool:
    """Проверяет, пуста ли папка. Если не существует — создаёт."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True
    return len(os.listdir(folder_path)) == 0


def parse_cian():
    """Парсинг квартир по числу комнат с сохранением CSV в data/raw"""
    if not is_folder_empty(RAW_DATA_DIR):
        print(f"Папка {RAW_DATA_DIR} не пуста, парсинг пропускается.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    for n_rooms in range(1, 6):  # от 1 до 5 комнат
        try:
            print(f"Парсинг квартир с {n_rooms} комнатой(ами)...")
            csv_path = f"{RAW_DATA_DIR}/{n_rooms}room_{timestamp}.csv"
            data = moscow_parser.get_flats(
                deal_type="sale",
                rooms=(n_rooms,),
                with_saving_csv=False,
                additional_settings={
                    "start_page": 1,
                    "end_page": 20,
                    "object_type": "secondary"
                }
            )
            if data:
                df = pd.DataFrame(data)
                df.to_csv(csv_path, encoding='utf-8', index=False)
                print(f"Успешно сохранено в {csv_path}")
            else:
                print(f"Не найдено данных для {n_rooms} комнат")
        except Exception as e:
            print(f"Ошибка при парсинге {n_rooms} комнат: {str(e)}")


def preprocess_data() -> pd.DataFrame | None:
    """Предобработка: объединение CSV, фильтрация и сохранение в progress_data.csv"""
    file_list = glob.glob(f"{RAW_DATA_DIR}/*.csv")
    if not file_list:
        print("В папке raw нет CSV файлов для обработки.")
        return None

    try:
        # Объединение всех CSV-файлов
        main_df = pd.concat((pd.read_csv(file) for file in file_list), axis=0)
        main_df['url_id'] = main_df['url'].map(lambda x: x.split('/')[-2])
        filtered_df = main_df[['url_id', 'total_meters', 'price']].set_index('url_id')

        # Фильтрация: исключение выбросов
        filtered_df = filtered_df[
            (filtered_df['price'] < 100_000_000) &
            (filtered_df['total_meters'] > 10) &
            (filtered_df['total_meters'] < 120)
        ]

        # Создание директории и сохранение
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        filtered_df.to_csv(PROCESSED_DATA_PATH)
        print(f"Данные успешно сохранены в файл: {PROCESSED_DATA_PATH}")
        return filtered_df
    except Exception as e:
        print(f"Ошибка при предобработке данных: {str(e)}")
        return None


def train_model():
    """Обучение модели LinearRegression и сохранение"""
    data = pd.read_csv(PROCESSED_DATA_PATH)

    X = data[['total_meters']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Метрики
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Средняя ошибка: {np.mean(np.abs(y_test - y_pred)):.2f} руб.")

    print("\nКоэффициенты регрессии:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"{name}: {coef:.2f}")
    print(f"Свободный член (intercept): {model.intercept_:.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в файл {MODEL_PATH}")


def test_model():
    """Запуск сервиса Flask для предсказаний"""
    # Логгирование
    dictConfig({
        "version": 1,
        "formatters": {
            "default": {"format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
            "file": {"class": "logging.FileHandler", "filename": "service/flask.log", "formatter": "default"}
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]}
    })

    app = Flask(__name__)

    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info("Модель успешно загружена")
    except Exception as e:
        app.logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/numbers', methods=['POST'])
    def process_numbers():
        try:
            data = request.get_json()
            app.logger.info(f"Request data: {data}")

            area = float(data['area'])
            rooms = int(data['rooms'])
            total_floors = int(data['total_floors'])
            floor = int(data['floor'])

            # Валидация
            if area <= 0 or rooms <= 0 or total_floors <= 0 or floor <= 0:
                return {'status': 'error', 'message': 'Все значения должны быть положительными'}, 400
            if floor > total_floors:
                return {'status': 'error', 'message': 'Этаж не может быть выше количества этажей'}, 400

            prediction = model.predict([[area]])[0]
            return {
                'status': 'success',
                'data': {
                    'estimated_price': prediction,
                    'price_per_m2': round(prediction / area),
                    'parameters': {
                        'area': area,
                        'rooms': rooms,
                        'total_floors': total_floors,
                        'floor': floor
                    },
                    'model_used': True
                }
            }
        except (ValueError, KeyError) as e:
            app.logger.error(f"Ошибка валидации: {e}")
            return {'status': 'error', 'message': 'Некорректные данные'}, 400
        except Exception as e:
            app.logger.error(f"Ошибка модели: {e}")
            return {'status': 'error', 'message': 'Ошибка предсказания'}, 500

    app.run(debug=True)


if __name__ == "__main__":
    # CLI интерфейс
    parser = argparse.ArgumentParser(description="Полный цикл работы ML модели")
    parser.add_argument(
        "-m", "--model", default=MODEL_PATH, help="Путь к файлу модели"
    )
    parser.add_argument(
        "--run-server", action="store_true", help="Запустить Flask сервер"
    )
    parser.add_argument(
        "--train", action="store_true", help="Обучить модель"
    )

    args = parser.parse_args()

    parse_cian()
    preprocess_data()
    train_model()
    test_model()

    '''if args.train:
        train_model()

    if args.run_server:
        test_model()'''