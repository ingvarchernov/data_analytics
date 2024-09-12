import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram_bot.pair_handler import PairHandler
from telegram_bot.indicator_handler import IndicatorHandler
from config import TELEGRAM_TOKEN
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await PairHandler.select_pair(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ви можете використовувати наступні команди:\n"
        "/start - Почати роботу з ботом\n"
        "/analytics - Отримати технічний аналіз криптовалют"
    )


def main():
    # Створюємо екземпляр додатка
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Додаємо обробники команд та натискань
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(PairHandler.handle_pair_selection))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, IndicatorHandler.handle_indicator_selection))

    # Запускаємо бота
    application.run_polling()


if __name__ == '__main__':
    main()
