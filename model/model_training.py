import numpy as np
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api import Sequential, layers
from bi_api.data_extraction import get_historical_data
from indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)

def train_lstm_model(symbol='BTCUSDT', look_back=60, epochs=50, batch_size=32):
    logger.info(f"Завантаження даних для {symbol}.")
    data = get_historical_data(symbol)

    if data is None or data.empty:
        logger.error(f"Дані для {symbol} не були завантажені.")
        return None

    # Розрахунок індикаторів
    logger.info(f"Розрахунок технічних індикаторів для {symbol}.")
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['close'])
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    data['Stoch'], data['Stoch_Signal'] = calculate_stochastic(data)

    data = data.dropna()

    if data.empty:
        logger.error(f"Після розрахунку індикаторів немає достатньо даних для {symbol}.")
        return None

    logger.info("Масштабування даних.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close', 'RSI', 'MACD', 'MACD_Signal']].values)

    logger.info("Формування навчальних даних для моделі LSTM.")
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i - look_back:i])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    if X_train.shape[0] == 0:
        logger.error("Недостатньо даних для тренування.")
        return None

    logger.info("Створення та навчання моделі LSTM.")
    model = Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.LSTM(units=50))
    model.add(layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    model.save(f'{symbol}_lstm_model.h5')
    logger.info(f"Модель для {symbol} успішно збережено.")

    plot_training_results(history, symbol)

    return model

def plot_training_results(history, symbol):
    logger.info(f"Побудова графіків втрат для {symbol}.")
    plt.plot(history.history['loss'], label='Втрати на навчальних даних')
    plt.plot(history.history['val_loss'], label='Втрати на валідаційних даних')
    plt.title(f'Втрати під час тренування для {symbol}')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати (MSE)')
    plt.legend()
    plt.show()
