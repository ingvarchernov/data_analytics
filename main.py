import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
)
from telegram_bot.training_handler import TrainingHandler
from telegram_bot.indicator_handler import IndicatorHandler
from config import TELEGRAM_TOKEN

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define stages for ConversationHandler
START_ROUTES, END_ROUTES = range(2)

# Callback data
AI_MODEL, TECH_INDICATORS, START_TRAINING, CHOOSE_INDICATORS, ALL_INDICATORS, BACK_TO_START, SELECT_CURRENCY_PAIR = range(7)
CURRENCY_PAIRS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT"]  # Надаємо стану унікальне число


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /start з кнопками 'AI model' та 'Tech Indicators'."""
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)

    # Кнопки для початкового меню
    keyboard = [
        [InlineKeyboardButton("AI model", callback_data=str(AI_MODEL))],
        [InlineKeyboardButton("Tech Indicators", callback_data=str(TECH_INDICATORS))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("Ласкаво просимо! Оберіть одну з опцій нижче:", reply_markup=reply_markup)
    return START_ROUTES

async def show_ai_model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню для 'AI model' з кнопками 'Start Training' та 'Назад'."""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("Start Training", callback_data=str(START_TRAINING))],
        [InlineKeyboardButton("Назад", callback_data=str(BACK_TO_START))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text("Оберіть дію для AI моделі:", reply_markup=reply_markup)
    return START_ROUTES

async def show_tech_indicators_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню для 'Tech Indicators' з кнопками 'Choose' та 'All'."""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("Choose", callback_data=str(CHOOSE_INDICATORS))],
        [InlineKeyboardButton("All", callback_data=str(ALL_INDICATORS))],
        [InlineKeyboardButton("Назад", callback_data=str(BACK_TO_START))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text("Оберіть дію для технічних індикаторів:", reply_markup=reply_markup)
    return START_ROUTES

@staticmethod
async def select_currency_pair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробка вибору валютної пари."""
    query = update.callback_query
    await query.answer()
    selected_pair = query.data  # Зберігаємо вибрану валютну пару
    context.user_data['selected_pair'] = selected_pair  # Зберігаємо в user_data

    await query.edit_message_text(f"Ви обрали валютну пару: {selected_pair}. Тепер оберіть індикатори.")
    # Переходимо до вибору індикаторів
    await show_tech_indicators_menu(update, context)  # Виклик функції для відображення меню індикаторів
    return START_ROUTES

async def show_currency_pair_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню для вибору валютних пар."""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("BTC/USDT", callback_data="BTC/USDT")],
        [InlineKeyboardButton("ETH/USDT", callback_data="ETH/USDT")],
        [InlineKeyboardButton("XRP/USDT", callback_data="XRP/USDT")],
        [InlineKeyboardButton("LTC/USDT", callback_data="LTC/USDT")],
        [InlineKeyboardButton("Назад", callback_data=str(BACK_TO_START))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text("Оберіть валютну пару для аналізу:", reply_markup=reply_markup)
    return SELECT_CURRENCY_PAIR

async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Оновлення початкового меню."""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("AI model", callback_data=str(AI_MODEL))],
        [InlineKeyboardButton("Tech Indicators", callback_data=str(TECH_INDICATORS))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text("Ласкаво просимо! Оберіть одну з опцій нижче:", reply_markup=reply_markup)
    return START_ROUTES

async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Закінчення розмови."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text="До зустрічі наступного разу!")
    return ConversationHandler.END

def main() -> None:
    """Запуск бота."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # ConversationHandler для управління потоками вибору меню
    conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        START_ROUTES: [
            CallbackQueryHandler(show_ai_model_menu, pattern="^" + str(AI_MODEL) + "$"),
            CallbackQueryHandler(TrainingHandler.handle_ai_model_selection, pattern="^" + str(START_TRAINING) + "$"),
            CallbackQueryHandler(show_currency_pair_menu, pattern="^" + str(TECH_INDICATORS) + "$"),  # Змінюємо на вибір пар
        ],
         SELECT_CURRENCY_PAIR: [
                CallbackQueryHandler(select_currency_pair, pattern="^(" + "|".join(CURRENCY_PAIRS) + ")$"),  # Додаємо обробку валютних пар
            ],
        START_ROUTES: [
            CallbackQueryHandler(IndicatorHandler.handle_tech_indicators_selection, pattern="^" + str(CHOOSE_INDICATORS) + "$"),
            CallbackQueryHandler(IndicatorHandler.handle_tech_indicators_selection, pattern="^" + str(ALL_INDICATORS) + "$"),
            CallbackQueryHandler(start_over, pattern="^" + str(BACK_TO_START) + "$"),
        ],
        END_ROUTES: [
            CallbackQueryHandler(start_over, pattern="^" + str(BACK_TO_START) + "$"),
            CallbackQueryHandler(end, pattern="^" + str(END_ROUTES) + "$"),
        ],
    },
    fallbacks=[CommandHandler("start", start)],
)


    # Додаємо ConversationHandler до додатку
    application.add_handler(conv_handler)

    # Запускаємо бота
    application.run_polling()

if __name__ == "__main__":
    main()
