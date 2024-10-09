import logging
from telegram import Update
from telegram.ext import ContextTypes
from model.model_training import train_lstm_model  # Імпортуємо функцію тренування моделі

logger = logging.getLogger(__name__)

class TrainingHandler:
   @staticmethod
   async def start_lstm_training(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск тренування LSTM моделі."""
        query = update.callback_query
        await query.answer()

        # Отримання символу з вибраної пари
        selected_pair = context.user_data.get('selected_pair', 'BTCUSDT')

        await query.edit_message_text(f"Починаємо тренування моделі LSTM для {selected_pair}...")
        try:
            # Запуск процесу тренування моделі
            result = train_lstm_model(symbol=selected_pair)

            if result:
                await query.edit_message_text("Тренування завершено успішно!")
            else:
                await query.edit_message_text("Сталася помилка під час тренування.")
        except Exception as e:
            logger.error(f"Помилка під час тренування LSTM: {e}")
            await query.edit_message_text(f"Помилка під час тренування: {e}")

