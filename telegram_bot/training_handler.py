import logging
from telegram import Update
from telegram.ext import ContextTypes
from model.model_training import train_lstm_model  # Імпортуємо функцію тренування моделі

logger = logging.getLogger(__name__)

class TrainingHandler:
   @staticmethod
   async def start_lstm_training(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск тренування LSTM моделі."""
        await update.message.reply_text("Починаємо тренування моделі LSTM...")
        try:
            result = train_lstm_model(symbol=context.user_data.get('selected_pair'))
            if result:
                await update.message.reply_text("Тренування завершено успішно!")
            else:
                await update.message.reply_text("Сталася помилка під час тренування.")
        except Exception as e:
            logger.error(f"Помилка під час тренування LSTM: {e}")
            await update.message.reply_text(f"Помилка: {e}")

