"""This is full life cycle for ml model"""

import argparse
import datetime
import cianparser
import pandas as pd
import glob #для поиска файлов по шаблону
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from flask import Flask, render_template, request 
from logging.config import dictConfig
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


TRAIN_SIZE = 0.2
MODEL_NAME = "linear_regression_model_v2.pkl"
moscow_parser = cianparser.CianParser(location="Москва")


def is_folder_empty(folder_path):
    """Проверяет, пуста ли папка"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True
    return len(os.listdir(folder_path)) == 0

def parse_cian():
    """Parse data to data/raw"""

    raw_folder = 'data/raw'
    if not is_folder_empty(raw_folder):
        print(f"Папка {raw_folder} не пуста, парсинг пропускается.")
        return

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    for n_rooms in range(1, 6):  # 1, 2, 3, 4, 5
        try:
            print(f"Парсинг квартир с {n_rooms} комнатой(ами)...")
            csv_path = f'{raw_folder}/{n_rooms}room_{t}.csv'
            #csv_path = f'data/raw/{n_rooms}_{t}.csv'
            data = moscow_parser.get_flats(
                deal_type="sale",
                rooms=(n_rooms,),
                with_saving_csv=False,
                additional_settings={
                    "start_page": 1,
                    "end_page": 20,
                    "object_type": "secondary"
                })
            
            if data:  # Проверяем, что данные получены
                df = pd.DataFrame(data)
                df.to_csv(csv_path, encoding='utf-8', index=False)
                print(f"Успешно сохранено в {csv_path}")
            else:
                print(f"Не найдено данных для {n_rooms} комнат")
                
        except Exception as e:
            print(f"Ошибка при парсинге {n_rooms} комнат: {str(e)}")
    pass


def preprocess_data():
    """Filter and remove"""
    raw_data_path = 'data/raw'
    file_list = glob.glob(raw_data_path + "/*.csv") 

    # Проверяем, есть ли файлы для обработки
    if not file_list:
        print("В папке raw нет CSV файлов для обработки")
        return None

    try:
        # Создаем основной DataFrame, загружая первый файл из списка
        main_dataframe = pd.read_csv(file_list[0])

        # Загружаем остальные файлы и объединяем их с основным DataFrame
        for file_path in file_list[1:]: 
            data = pd.read_csv(file_path) 
            main_dataframe = pd.concat([main_dataframe, data], axis=0)

        # Создаем новый DataFrame с нужными столбцами
        # Извлекаем id из URL и устанавливаем его как индекс
        main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
        new_dataframe = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')

        # Определяем путь для сохранения файла
        output_path = '/home/eva/project/predictive_real_estate_price_analytics_service/data/progress/progress_data.csv'

        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Фильтруем данные
        filtered_df = new_dataframe[
            (new_dataframe['price'] < 100_000_000) &
            (new_dataframe['total_meters'] > 10) &
            (new_dataframe['total_meters'] < 120)
        ]

        # Сохраняем DataFrame в CSV файл
        filtered_df.to_csv(output_path, index=True)
        print(f"Данные успешно сохранены в файл: {output_path}")

        return filtered_df

    except Exception as e:
        print(f"Произошла ошибка при обработке данных: {str(e)}")
        return None


def train_model():
    """Train model and save with MODEL_NAME"""
    # Загрузка данных
    data = pd.read_csv('/home/eva/project/predictive_real_estate_price_analytics_service/data/progress/progress_data.csv')
    data.head()
    


    X = data[['total_meters']]  # только один признак - площадь
    y = data['price']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели CatBoost
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=100,  # выводит информацию каждые 100 итераций
        random_state=42
    )
    model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Вывод метрик качества
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    print(f"Коэффициент детерминации R²: {r2:.6f}")
    print(f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей")

    # Для CatBoost коэффициенты модели интерпретируются иначе, чем в линейной регрессии
    # Можно получить важность признаков
    print("\nВажность признаков:")
    for name, score in zip(X.columns, model.get_feature_importance()):
        print(f"{name}: {score:.2f}")
    

    # Сохранение модели
    model_path = '/home/eva/project/predictive_real_estate_price_analytics_service/models/linear_regression_model_v2.pkl'

    joblib.dump(model, model_path)
    print(f"Модель сохранена в файл {model_path}") 

    # Загрузка модели
    loaded_model = joblib.load(model_path)
    print("Модель загружена из файла")



    pass

#********************************************************************************************************************************************
def test_model():
    """Test model with new data"""
    dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },         
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
    )

    app = Flask(__name__)

    #************************************************
    # Загрузка обученной модели
    try:
        model = joblib.load('models/linear_regression_model_v2.pkl')
        app.logger.info("Модель успешно загружена")
    except Exception as e:
        app.logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise

    # Создание scaler (должен быть таким же, как при обучении)
    scaler = StandardScaler()

    #**********************************************************

    # Базовая цена за квадратный метр
    #BASE_PRICE_PER_M2 = 300000

    # Маршрут для отображения формы
    @app.route('/')
    def index():
        return render_template('index.html')

    # Маршрут для обработки данных формы
    @app.route('/api/numbers', methods=['POST'])
    def process_numbers():
        data = request.get_json()
        
        app.logger.info(f'Requst data: {data}')

        #___________________________________________________________________________
        try:
            # Получаем параметры из запроса
            area = float(data['area'])
            rooms = int(data['rooms'])
            total_floors = int(data['total_floors'])
            floor = int(data['floor'])
            
            # Проверяем корректность данных
            if area <= 0 or rooms <= 0 or total_floors <= 0 or floor <= 0:
                return {'status': 'error', 'message': 'Все значения должны быть положительными числами'}, 400
            
            if floor > total_floors:
                return {'status': 'error', 'message': 'Этаж квартиры не может быть больше общего количества этажей'}, 400
            
            #**************************************************************
            # Подготовка данных для модели
            # Масштабируем площадь так же, как при обучении
            #area_scaled = scaler.transform([[area]])[0][0]
            
            # Делаем предсказание с помощью модели
            predicted_price = model.predict([[area]])[0]
            print('predicted_price: ', predicted_price )
            return {
                'status': 'success', 
                'data': {
                    'estimated_price': predicted_price,
                    'price_per_m2': round(predicted_price / area),
                    'parameters': {
                        'area': area,
                        'rooms': rooms,
                        'total_floors': total_floors,
                        'floor': floor
                    },
                    'model_used': True  # Флаг, что использовалась ML модель
                }
            }
        
        except (ValueError, KeyError) as e:
            app.logger.error(f'Error processing request: {str(e)}')
            return {'status': 'error', 'message': 'Некорректные данные'}, 400
        except Exception as e:
            app.logger.error(f'Model prediction error: {str(e)}')
            return {'status': 'error', 'message': 'Ошибка предсказания модели'}, 500
                
        #return {'status': 'success', 'data': 'Числа успешно обработаны'}
    app.run(debug=True)
    pass



if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()

    parse_cian()
    preprocess_data()
    #train_model()
    test_model()
