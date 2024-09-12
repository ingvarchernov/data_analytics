import logging
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def execute_trade(decision, symbol, quantity):
    logger.info(f"Виконання торгового рішення: {decision} {quantity} {symbol}")
    try:
        if decision == "BUY":
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
        elif decision == "SELL":
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logger.info(f"Виконано {decision} ордер: {order}")
    except Exception as e:
        logger.error(f"Помилка при виконанні торгівлі: {e}")
