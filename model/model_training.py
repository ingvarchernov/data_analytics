import logging
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

logger = logging.getLogger(__name__)


def create_model(input_shape):
    logger.info(f"Створення моделі з входом форми {input_shape}.")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info(f"Модель успішно створена.")
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=64):
    logger.info(f"Початок тренування моделі на {epochs} епохах з batch_size {batch_size}.")
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    logger.info(f"Тренування моделі завершено.")
    logger.debug(f"Історія тренувань: {history.history}")
    return model
