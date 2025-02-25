import logging
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_historical_data(symbol, interval='1h', days_back=90, max_per_request=1000):
    """
    Витягує історичні дані з Binance за вказаний період.

    Args:
        symbol (str): Торгова пара (наприклад, 'BTCUSDT').
        interval (str): Інтервал часу ('1h', '1d', тощо).
        days_back (int): Кількість днів назад від поточного часу (default: 90).
        max_per_request (int): Максимальна кількість записів за один запит (max 1000).

    Returns:
        pd.DataFrame: Оброблені історичні дані.
    """
    logger.info(f"Отримання історичних даних для {symbol} з інтервалом {interval}, за {days_back} днів.")

    # Визначення часового діапазону
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    start_timestamp = int(start_time.timestamp() * 1000)  # У мілісекундах
    end_timestamp = int(end_time.timestamp() * 1000)

    all_data = []
    current_start = start_timestamp

    try:
        while current_start < end_timestamp:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=max_per_request,
                startTime=current_start,
                endTime=end_timestamp
            )

            if not klines:
                logger.warning(f"Дані для {symbol} за період {current_start} порожні.")
                break

            all_data.extend(klines)

            # Оновлення початкової точки для наступного запиту
            last_timestamp = klines[-1][0]  # Час останньої свічки
            current_start = last_timestamp + 1  # Додаємо 1мс, щоб уникнути дублювання

            logger.debug(f"Отримано {len(klines)} записів, останній timestamp: {last_timestamp}")
            time.sleep(0.1)  # Затримка для уникнення перевищення ліміту API

        if not all_data:
            raise ValueError(f"Дані для {symbol} не отримано за {days_back} днів.")

        # Перетворення в DataFrame
        data = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])

        # Конвертація колонок
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Видалення дублікатів за timestamp
        data.drop_duplicates(subset=['timestamp'], inplace=True)

        # Очищення від NaN
        data.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        if data.empty:
            raise ValueError(f"Після очищення дані для {symbol} порожні.")

        # Логарифмування числових колонок для зменшення волатильності
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[f'log_{col}'] = np.log1p(data[col])

        logger.info(f"Дані успішно оброблені: {data.shape[0]} записів для {symbol}.")
        return data

    except Exception as e:
        logger.error(f"Помилка при отриманні даних для {symbol}: {e}")
        raise

def fetch_extended_data(symbol, interval='1h', start_date=None, end_date=None, max_per_request=1000):
    """
    Витягує дані за конкретний період часу з кількома запитами.

    Args:
        symbol (str): Торгова пара.
        interval (str): Інтервал часу.
        start_date (str): Початкова дата у форматі 'YYYY-MM-DD' (опціонально).
        end_date (str): Кінцева дата у форматі 'YYYY-MM-DD' (опціонально).
        max_per_request (int): Максимальна кількість записів за один запит.

    Returns:
        pd.DataFrame: Оброблені історичні дані.
    """
    if start_date:
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_time = datetime.now() - timedelta(days=90)

    if end_date:
        end_time = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_time = datetime.now()

    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)

    logger.info(f"Витягування даних для {symbol} з {start_date} до {end_date}.")
    return get_historical_data(symbol, interval, days_back=None, max_per_request=max_per_request)