import logging
from binance.client import Client
import pandas as pd
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def get_historical_data(symbol, interval='1h', limit=500):
    logger.info(f"Отримання історичних даних для {symbol} з інтервалом {interval} та обмеженням {limit} записів.")
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        logger.debug(f"Отримані дані: {klines[:5]}...")  # Логувати лише перші 5 записів для зручності

        if not klines:
            raise ValueError("Не вдалося отримати дані для символу {symbol}.")

        # Перетворення в DataFrame
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

        logger.debug(f"Тип даних: {type(data)}")
        logger.debug(f"Колонки в даних: {data.columns if isinstance(data, pd.DataFrame) else 'Дані не є DataFrame'}")
        logger.debug(f"Перші 5 записів: {data.head() if isinstance(data, pd.DataFrame) else data}")

        # Перетворення колонок у числовий формат
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        logger.info(f"Дані успішно оброблені для {symbol}.")
        return data
    except Exception as e:
        logger.error(f"Помилка при отриманні даних для {symbol}: {e}")
        raise


