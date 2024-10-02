from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from telegram_bot.indicator_handler import IndicatorHandler
<<<<<<< HEAD
from telegram_bot.training_handler import TrainingHandler
=======
>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211


class PairHandler:
    @staticmethod
    async def select_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати клавіатуру для вибору валютної пари"""
        keyboard = [
            [InlineKeyboardButton("BTC/USDT", callback_data='BTCUSDT')],
            [InlineKeyboardButton("ETH/USDT", callback_data='ETHUSDT')],
            [InlineKeyboardButton("LTC/USDT", callback_data='LTCUSDT')],
<<<<<<< HEAD
            [InlineKeyboardButton("BNB/USDT", callback_data='BNBUSDT')],
            [InlineKeyboardButton("Запустити тренування", callback_data='train_model')]
=======
            [InlineKeyboardButton("BNB/USDT", callback_data='BNBUSDT')]
>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Оберіть валютну пару:', reply_markup=reply_markup)

    @staticmethod
    async def handle_pair_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка вибору валютної пари та перехід до вибору індикаторів"""
        query = update.callback_query
        await query.answer()

<<<<<<< HEAD
         # Якщо вибрано тренування
        if query.data == 'train_model':
            await TrainingHandler.start_training(update, context)  # Запускаємо тренування моделі
            return

=======
>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211
        # Зберігаємо обрану пару
        selected_pair = query.data
        context.user_data['selected_pair'] = selected_pair

        # Оновлюємо повідомлення та показуємо вибір індикаторів
        await query.edit_message_text(f"Ви обрали пару {selected_pair}. Тепер оберіть технічні індикатори.")
        await IndicatorHandler.select_indicators(update, context)
