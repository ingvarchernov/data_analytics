import logging
import pandas as pd
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import numpy as np

logger = logging.getLogger(__name__)


def calculate_moving_average(data, window=20):
    logger.info(f"Розрахунок ковзного середнього з вікном {window}.")
    ma = data['close'].rolling(window=window).mean()
    logger.debug(f"Результат MA: {ma.tail()}")
    return ma


def calculate_rsi(data, window=14):
    logger.info(f"Розрахунок RSI з вікном {window}.")
    logger.debug(f"Тип даних для 'close': {type(data['close'])}")
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    logger.debug(f"Результат RSI: {rsi.tail()}")
    return rsi


def calculate_bollinger_bands(data, window=20):
    logger.info(f"Розрахунок смуг Боллінджера з вікном {window}.")
    sma = calculate_moving_average(data, window)
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    logger.debug(f"Верхня смуга: {upper_band.tail()}, Нижня смуга: {lower_band.tail()}")
    return upper_band, lower_band


def calculate_macd(closes):
    logger.info("Розрахунок MACD.")
    # Переконуємося, що closes є pandas.Series
    if not isinstance(closes, pd.Series):
        raise TypeError("`closes` must be a pandas Series.")

    macd = MACD(closes, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()

    logger.debug(f"MACD Line: {macd_line.tail()}, Signal Line: {signal_line.tail()}")
    return macd_line, signal_line


def calculate_stochastic(data, window=14):
    logger.info(f"Розрахунок Stochastic Oscillator з вікном {window}.")
    stochastic_indicator = StochasticOscillator(data['high'], data['low'], data['close'], window=window)
    stoch = stochastic_indicator.stoch()
    stoch_signal = stochastic_indicator.stoch_signal()
    logger.debug(f"Результат Stochastic Oscillator: {stoch.tail()}, Stochastic Signal: {stoch_signal.tail()}")
    return stoch, stoch_signal

def calculate_ema(data, window=14):
    logger.info(f"Розрахунок EMA з вікном {window}.")

    if not isinstance(data['close'], pd.Series):
        raise TypeError("`data['close']` must be a pandas Series.")

    ema = data['close'].ewm(span=window, adjust=False).mean()
    logger.debug(f"Результат EMA: {ema.tail()}")
    return ema

def calculate_atr(data, window=14):
    logger.info(f"Розрахунок ATR з вікном {window}.")

    # Перевіряємо, чи є потрібні стовпці в даних
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("`data` must contain 'high', 'low', and 'close' columns.")

    # Перетворення стовпця 'high' на числовий тип і видалення NaN
    data['high'] = pd.to_numeric(data['high'], errors='coerce')
    data['low'] = pd.to_numeric(data['low'], errors='coerce')
    data['close'] = pd.to_numeric(data['close'], errors='coerce')

    # Логування та перевірка на наявність некоректних даних
    if data[['high', 'low', 'close']].isnull().any().any():
        #logger.error(f"Наявність NaN після конвертації в стовпцях 'high', 'low', або 'close':\n{data[['high', 'low', 'close']].isnull().sum()}")
        raise ValueError("Дані містять NaN значення після перетворення на числовий формат.")

    data.dropna(subset=['high', 'low', 'close'], inplace=True)

    if data.empty:
        logger.error("Після видалення NaN даних недостатньо для розрахунку ATR.")
        raise ValueError("Недостатньо даних після очищення для розрахунку ATR.")

    # Обчислення True Range
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())

    # Логування результатів перед обчисленням максимального значення
    logger.debug(f"high_low:\n{high_low.tail()}")
    logger.debug(f"high_close:\n{high_close.tail()}")
    logger.debug(f"low_close:\n{low_close.tail()}")

    true_range = pd.DataFrame({
        'high_low': high_low,
        'high_close': high_close,
        'low_close': low_close
    })

    true_range['tr'] = true_range.max(axis=1)  # Обчислення True Range як максимальне з трьох значень

    # Обчислення ATR як ковзного середнього True Range
    atr = true_range['tr'].rolling(window=window).mean()

    logger.info(f"Результат ATR: {atr.tail()}")
    return atr


def calculate_cci(data, window=20):
    logger.info(f"Розрахунок CCI з вікном {window}.")

    if not pd.api.types.is_numeric_dtype(data['high']):
        logger.info(f"Значення ссi 'high': {data['high'].head()}")
        raise ValueError("The cci 'high' column must contain numeric values.")

    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("`data` must contain 'high', 'low', and 'close' columns.")

    tp = (data['high'] + data['low'] + data['close']) / 3  # Типова ціна
    sma = tp.rolling(window=window).mean()  # Проста ковзна середня
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))  # Середнє відхилення
    cci = (tp - sma) / (0.015 * mad)

    logger.debug(f"Результат CCI: {cci.tail()}")
    return cci

def calculate_obv(data):
    logger.info("Розрахунок OBV.")

    # Перетворення 'volume' на числовий формат
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

    # Видалення рядків з NaN після конвертації
    data.dropna(subset=['volume'], inplace=True)

    if data.empty:
        logger.error("Недостатньо даних для розрахунку OBV після очищення 'volume'.")
        raise ValueError("Недостатньо даних для розрахунку OBV.")

    # Перевіряємо, чи всі стовпці є в даних
    if not all(col in data.columns for col in ['close', 'volume']):
        raise ValueError("`data` must contain 'close' and 'volume' columns.")

    # Початкове значення OBV
    obv = [0]

    # Обчислення OBV на основі змін закриття та обсягу
    for i in range(1, len(data['close'])):
        if data['close'][i] > data['close'][i - 1]:
            obv.append(obv[-1] + data['volume'][i])
        elif data['close'][i] < data['close'][i - 1]:
            obv.append(obv[-1] - data['volume'][i])
        else:
            obv.append(obv[-1])

    # Перетворюємо список в Series для зручності роботи з Pandas
    obv_series = pd.Series(obv, index=data.index)

    logger.debug(f"Результат OBV: {obv_series.tail()}")
    return obv_series


