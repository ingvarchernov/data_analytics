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
        logger.info(f"Selected pair for training: {selected_pair}")

        await query.edit_message_text(f"Починаємо тренування моделі LSTM для {selected_pair}...")
        try:
            # Запуск процесу тренування моделі
            try:
                logger.info(f"Запуск тренування для символу: {selected_pair}")
                history, filename = train_lstm_model(symbol=selected_pair)
                logger.info(f"Тренування завершено для символу: {selected_pair}")
            except Exception as e:
                logger.error(f"Помилка виклику train_lstm_model: {e}")
                raise

            if history is not None:
                # Збереження історії тренування до Excel
                for fold_index, histories in enumerate(history):  # Assuming history contains multiple folds
                    export_training_history(histories, fold_index, filename)

                logger.info(f"Відправляємо файл {filename} користувачеві.")

                # Відправка Excel файлу в Telegram
                with open(filename, 'rb') as file:
                    await query.message.reply_document(InputFile(file, filename=filename))

                await query.edit_message_text("Тренування завершено успішно! Excel файл збережено.")
            else:
                await query.edit_message_text("Сталася помилка під час тренування.")
        except Exception as e:
            logger.error(f"Помилка під час тренування LSTM: {e}")
            await query.edit_message_text(f"Помилка під час тренування: {e}")

    @staticmethod
    async def handle_ai_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        from main import START_ROUTES, END_ROUTES
        """Обробка вибору з меню 'AI model'."""
        query = update.callback_query
        await query.answer()

        logger.info(f"Callback query data: {query.data}")  # Додаємо логування


        if  query.data == str(2):  # START_TRAINING
            # Викликаємо тренування моделі
            await TrainingHandler.start_lstm_training(update, context)
            logger.info(f"Вибираємо розпочати тренування {query}")
            return END_ROUTES  # Переходимо до кінця розмови після тренування

        return ConversationHandler.END  # Завершуємо розмову, якщо не співпадає
