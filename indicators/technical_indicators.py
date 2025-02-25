import logging
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
import numpy as np

logger = logging.getLogger(__name__)

def validate_data(data, required_columns):
    """Перевірка наявності колонок і непорожності даних."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Дані повинні бути непорожнім pandas DataFrame.")
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Відсутні колонки: {missing}")

def calculate_moving_average(data, window=20):
    logger.info(f"Розрахунок ковзного середнього з вікном {window}.")
    validate_data(data, ['close'])
    ma = data['close'].rolling(window=window).mean()
    logger.debug(f"MA статистика: mean={ma.mean():.2f}, std={ma.std():.2f}")
    return ma

def calculate_rsi(data, window=14):
    logger.info(f"Розрахунок RSI з вікном {window}.")
    validate_data(data, ['close'])
    rsi = RSIIndicator(data['close'], window=window).rsi()
    logger.debug(f"RSI статистика: mean={rsi.mean():.2f}, std={rsi.std():.2f}")
    return rsi

def calculate_bollinger_bands(data, window=20):
    logger.info(f"Розрахунок смуг Боллінджера з вікном {window}.")
    validate_data(data, ['close'])
    sma = calculate_moving_average(data, window)
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    logger.debug(f"Upper Band mean={upper_band.mean():.2f}, Lower Band mean={lower_band.mean():.2f}")
    return upper_band, lower_band

def calculate_macd(closes):
    logger.info("Розрахунок MACD.")
    if not isinstance(closes, pd.Series):
        raise TypeError("`closes` must be a pandas Series.")
    macd = MACD(closes, window_slow=26, window_fast=12, window_sign=9)
    macd_line, signal_line = macd.macd(), macd.macd_signal()
    logger.debug(f"MACD mean={macd_line.mean():.2f}, Signal mean={signal_line.mean():.2f}")
    return macd_line, signal_line

def calculate_stochastic(data, window=14):
    logger.info(f"Розрахунок Stochastic Oscillator з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    stochastic = StochasticOscillator(data['high'], data['low'], data['close'], window=window)
    stoch, stoch_signal = stochastic.stoch(), stochastic.stoch_signal()
    logger.debug(f"Stoch mean={stoch.mean():.2f}, Stoch Signal mean={stoch_signal.mean():.2f}")
    return stoch, stoch_signal

def calculate_ema(data, window=14):
    logger.info(f"Розрахунок EMA з вікном {window}.")
    validate_data(data, ['close'])
    ema = data['close'].ewm(span=window, adjust=False).mean()
    logger.debug(f"EMA mean={ema.mean():.2f}")
    return ema

def calculate_atr(data, window=14):
    logger.info(f"Розрахунок ATR з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    data = data.astype({'high': float, 'low': float, 'close': float}).dropna()
    if data.empty:
        raise ValueError("Недостатньо даних після очищення для ATR.")
    tr = pd.concat([
        data['high'] - data['low'],
        (data['high'] - data['close'].shift()).abs(),
        (data['low'] - data['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    logger.debug(f"ATR mean={atr.mean():.2f}")
    return atr

def calculate_cci(data, window=20):
    logger.info(f"Розрахунок CCI з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma) / (0.015 * mad)
    logger.debug(f"CCI mean={cci.mean():.2f}")
    return cci

def calculate_obv(data):
    logger.info("Розрахунок OBV.")
    validate_data(data, ['close', 'volume'])
    data = data.astype({'volume': float, 'close': float}).dropna()
    if data.empty:
        raise ValueError("Недостатньо даних для OBV.")
    direction = np.sign(data['close'].diff())
    obv = (direction * data['volume']).cumsum().fillna(0)
    logger.debug(f"OBV mean={obv.mean():.2f}")
    return obv