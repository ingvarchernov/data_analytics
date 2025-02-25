from telegram import Update, InputFile
from telegram.ext import ContextTypes, ConversationHandler
from model.model_training import train_lstm_model, export_training_history
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class TrainingHandler:
    @staticmethod
    async def start_lstm_training(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Логіка тренування LSTM моделі."""
        query = update.callback_query
        await query.answer()

        # Отримання символу з вибраної пари
        selected_pair = context.user_data.get('selected_pair', 'BTCUSDT')
        logger.info(f"Вибрана пара для тренування: {selected_pair}")

        await query.edit_message_text(f"Починаємо тренування моделі LSTM для {selected_pair}...")
        try:
            # Запуск процесу тренування моделі
            logger.info(f"Запуск тренування для символу: {selected_pair}")
            histories, filename = train_lstm_model(symbol=selected_pair)
            logger.info(f"Тренування завершено для символу: {selected_pair}")

            if histories is not None:
                # Збереження історії тренування до Excel (зазвичай це вже зроблено в cross_validate_lstm)
                # Але якщо потрібно викликати export_training_history окремо:
                for fold_index, history in enumerate(histories):
                    export_training_history(history, fold_index, filename)

                logger.info(f"Відправляємо файл {filename} користувачеві.")
                with open(filename, 'rb') as file:
                    await query.message.reply_document(InputFile(file, filename=filename))

                await query.edit_message_text("Тренування завершено успішно! Файл із результатами надіслано.")
            else:
                await query.edit_message_text("Сталася помилка під час тренування: дані не отримано.")
        except Exception as e:
            logger.error(f"Помилка під час тренування LSTM: {e}")
            await query.edit_message_text(f"Помилка під час тренування: {str(e)}")

    @staticmethod
    async def handle_ai_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        from main import START_ROUTES, END_ROUTES
        """Обробка вибору з меню 'AI model'."""
        query = update.callback_query
        await query.answer()

        logger.info(f"Дані зворотного виклику: {query.data}")

        if query.data == "2":  # START_TRAINING (str(2) для консистентності з main.py)
            await TrainingHandler.start_lstm_training(update, context)
            logger.info("Тренування моделі розпочато.")
            return END_ROUTES  # Завершуємо розмову після тренування

        return ConversationHandler.END  # Завершуємо, якщо вибір не співпадає