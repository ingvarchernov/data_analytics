import logging
import numpy as np

logger = logging.getLogger(__name__)


def make_trade_decision(model, current_data):
    logger.info(f"Прийняття торгового рішення на основі поточних даних.")
    X_input = np.reshape(current_data, (1, current_data.shape[0], 1))
    prediction = model.predict(X_input)
    logger.debug(f"Прогноз: {prediction}")

    if prediction > 0:
        logger.info("Рішення: BUY")
        return "BUY"
    elif prediction < 0:
        logger.info("Рішення: SELL")
        return "SELL"
    else:
        logger.info("Рішення: HOLD")
        return "HOLD"
