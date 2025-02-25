import numpy as np
import logging
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic, calculate_ema, calculate_atr, calculate_cci, calculate_obv
from bi_api.data_extraction import get_historical_data
import tensorflow as tf
import pandas as pd

logger = logging.getLogger(__name__)

def train_lstm_model(symbol='BTCUSDT', look_back=180, epochs=100, batch_size=64, n_splits=3):
    logger.info(f"Завантаження даних для {symbol}.")
    data = get_historical_data(symbol, interval='1h', days_back=180)

    if data is None or data.empty:
        logger.error(f"Дані для {symbol} не завантажено.")
        return None, None

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
        logger.error(f"Після розрахунку індикаторів даних недостатньо для {symbol}.")
        return None, None

    logger.info("Масштабування даних.")
    features = ['log_open', 'log_high', 'log_low', 'log_close', 'log_volume',
                'quote_av', 'trades', 'RSI', 'MACD', 'MACD_Signal',
                'Upper_Band', 'Lower_Band', 'Stoch', 'Stoch_Signal', 'EMA', 'ATR', 'CCI', 'OBV']
    scalers = {feat: MinMaxScaler(feature_range=(0, 1)) for feat in features}
    scaled_data = np.column_stack([scalers[feat].fit_transform(data[[feat]]) for feat in features])

    logger.info(f"Форма масштабованих даних: {scaled_data.shape}")
    logger.info(f"Діапазон 'log_close' після масштабування: {scaled_data[:, 3].min()} - {scaled_data[:, 3].max()}")

    logger.info("Формування навчальних даних для моделі LSTM.")
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 3])  # Індекс 3 — 'log_close'

    X, y = np.array(X), np.array(y)
    if X.shape[0] == 0:
        logger.error("Недостатньо даних для тренування.")
        return None, None

    logger.info(f"Форма X: {X.shape}, форма y: {y.shape}")

    logger.info("Початок крос-валідації та тренування.")
    histories, filename = cross_validate_lstm(X, y, epochs, batch_size, n_splits, scalers, data, features)

    if not histories:
        logger.error("Крос-валідація не повернула жодної історії тренування.")
        return None, None

    return histories, filename

def cross_validate_lstm(X, y, epochs, batch_size, n_splits, scalers, data, features):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    histories = []
    filename = 'training_history.xlsx'

    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model_with_attention(X_train.shape[1:])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=1)

        val_predictions = model.predict(X_val)
        real_preds = inverse_transform_predictions(val_predictions, scalers, data, features)
        real_y_val = inverse_transform_predictions(y_val, scalers, data, features)
        real_mae = np.mean(np.abs(real_preds - real_y_val))
        logger.info(f"Fold {fold} - Реальний MAE у одиницях BTC/USDT: {real_mae:.2f}")

        histories.append(history)

        history_df = pd.DataFrame(history.history)
        history_df['real_mae'] = real_mae
        sheet_name = f'Fold_{fold}'
        try:
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                history_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
                history_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Модель для Fold {fold} завершена.")
        fold += 1

    model.save(f'lstm_model_fold_{fold-1}.keras')
    return histories, filename

def build_model_with_attention(input_shape):
    inputs = layers.Input(shape=input_shape)
    lstm_1 = layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
    lstm_2 = layers.LSTM(384, return_sequences=True, recurrent_dropout=0.3,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005))(lstm_1)
    lstm_3 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(lstm_2)
    attention_out = layers.Attention()([lstm_3, lstm_3])
    pool_out = layers.GlobalAveragePooling1D()(attention_out)
    dense_1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(pool_out)
    dropout_1 = layers.Dropout(0.3)(dense_1)
    dense_2 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(dropout_1)
    dropout_2 = layers.Dropout(0.3)(dense_2)
    outputs = layers.Dense(1, activation='linear')(dropout_2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss='mean_squared_error', metrics=['mae'])
    return model

def inverse_transform_predictions(predictions, scalers, data, features):
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, 3] = predictions.flatten()  # Індекс 3 — 'log_close'
    scaled_back = np.column_stack([scalers[feat].inverse_transform(dummy[:, [i]]) for i, feat in enumerate(features)])
    real_preds = np.expm1(scaled_back[:, 3])  # Зворотнє логарифмування
    return real_preds

def export_training_history(history, fold_index, filename):
    history_df = pd.DataFrame(history.history)
    sheet_name = f'Fold_{fold_index + 1}'
    try:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            history_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            history_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Історію тренування для Fold {fold_index + 1} збережено до {filename}")