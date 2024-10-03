import numpy as np
import logging
import matplotlib.pyplot as plt
from keras.api import Sequential, layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic, calculate_ema, calculate_atr, calculate_cci, calculate_obv
from bi_api.data_extraction import get_historical_data
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

logger = logging.getLogger(__name__)

def train_lstm_model(symbol='BTCUSDT', look_back=60, epochs=550, batch_size=32, n_splits=5):
    logger.info(f"Завантаження даних для {symbol}.")
    data = get_historical_data(symbol)

    if data is None or data.empty:
        logger.error(f"Дані для {symbol} не були завантажені.")
        return None

    # Розрахунок технічних індикаторів
    logger.info(f"Розрахунок технічних індикаторів для {symbol}.")
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['close'])
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    data['Stoch'], data['Stoch_Signal'] = calculate_stochastic(data)
    data['EMA'] = calculate_ema(data)
    data['ATR'] = calculate_atr(data)
    data['CCI'] = calculate_cci(data)
    data['OBV'] = calculate_obv(data)

    data = data.dropna()

    if data.empty:
        logger.error(f"Після розрахунку індикаторів немає достатньо даних для {symbol}.")
        return None

    logger.info("Масштабування даних.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close', 'RSI', 'MACD', 'MACD_Signal', 'EMA', 'ATR', 'CCI', 'OBV']].values)

    logger.info("Формування навчальних даних для моделі LSTM.")
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0:
        logger.error("Недостатньо даних для тренування.")
        return None

    logger.info("Початок крос-валідації та тренування.")
    cross_validate_lstm(X, y, epochs, batch_size, n_splits)

    return None

def cross_validate_lstm(X, y, epochs, batch_size, n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    fold = 1
    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Побудова моделі
        model = build_model_with_attention(X_train.shape[1:])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

        logger.info(f"Модель для Fold {fold} завершена.")
        plot_training_results(history, f"Fold {fold}")
        fold += 1

def build_model_with_attention(input_shape):
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(units=50, return_sequences=True), input_shape=input_shape))
    model.add(layers.LSTM(units=50, return_sequences=True))

    # Додавання Attention шару
    model.add(layers.Attention())

    model.add(layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_training_results(history, title):
    logger.info(f"Побудова графіків втрат для {title}.")
    plt.plot(history.history['loss'], label='Втрати на навчальних даних')
    plt.plot(history.history['val_loss'], label='Втрати на валідаційних даних')
    plt.title(f'Втрати під час тренування для {title}')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати (MSE)')
    plt.legend()
    plt.show()

# Гіперпараметричний пошук з Keras Tuner
def hyperparameter_search(X_train, y_train, epochs):
    def model_builder(hp):
        model = keras.Sequential()
        hp_units = hp.Int('units', min_value=32, max_value=128, step=16)
        model.add(keras.layers.LSTM(units=hp_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))

        hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(hp_dropout))

        model.add(keras.layers.Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=3
    )

    tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model
