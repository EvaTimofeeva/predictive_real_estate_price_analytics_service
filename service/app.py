from flask import Flask, render_template, request 
from logging.config import dictConfig
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    model = joblib.load('models/linear_regression_model.pkl')
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
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно
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
        predicted_price = np.expm1(model.predict([[area]]))[0]
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
    
        #***********************************************************


        """

        # Вычисляем стоимость квартиры
        # Базовая стоимость: площадь * базовая цена
        price = area * BASE_PRICE_PER_M2
        
        # Надбавка за количество комнат (5% за каждую комнату сверх 1)
        if rooms > 1:
            price *= 1 + 0.05 * (rooms - 1)
        
        # Надбавка за этаж (если не первый и не последний)
        if 1 < floor < total_floors:
            price *= 1.1  # +10% за "хороший" этаж
        elif floor == total_floors:
            price *= 0.95  # -5% за последний этаж
        else:
            price *= 0.9   # -10% за первый этаж
        
        # Округляем до тысяч
        price = round(price / 1000) * 1000
        
        return {
            'status': 'success', 
            'data': {
                'estimated_price': price,
                'price_per_m2': round(price / area),
                'parameters': {
                    'area': area,
                    'rooms': rooms,
                    'total_floors': total_floors,
                    'floor': floor
                }
            }
        }

        
    
    except (ValueError, KeyError) as e:
        app.logger.error(f'Error processing request: {str(e)}')
        return {'status': 'error', 'message': 'Некорректные данные'}, 400 """
    
    #return {'status': 'success', 'data': 'Числа успешно обработаны'}

if __name__ == '__main__':
    app.run(debug=True)
