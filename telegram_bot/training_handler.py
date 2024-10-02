import logging
from telegram import Update
from telegram.ext import ContextTypes
from model.model_training import train_lstm_model  # Імпортуємо функцію тренування моделі

logger = logging.getLogger(__name__)

class TrainingHandler:
    @staticmethod
    async def start_training(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск процесу тренування моделі та виведення результатів"""
        query = update.callback_query  # Отримуємо callback query
        await query.edit_message_text("Тренування моделі розпочато. Це може зайняти деякий час...")  # Відповідаємо на callback

        try:
            result = train_lstm_model()  # Запускаємо тренування моделі
            await query.edit_message_text(f"Тренування завершено. Результати: {result}")
        except Exception as e:
            logger.error(f"Помилка під час тренування: {str(e)}")
            await query.edit_message_text(f"Виникла помилка під час тренування: {str(e)}")
