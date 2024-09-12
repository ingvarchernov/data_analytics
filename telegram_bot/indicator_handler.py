import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import logging
from telegram import InputFile, ReplyKeyboardMarkup
from telegram import Update
from telegram.ext import ContextTypes
from bi_api.data_extraction import get_historical_data
from indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, \
    calculate_stochastic

logger = logging.getLogger(__name__)


class IndicatorHandler:
    @staticmethod
    async def select_indicators(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати клавіатуру для вибору технічних індикаторів після вибору пари"""
        indicators = [["RSI", "MACD"], ["Bollinger Bands", "Stochastic"], ["Усі"]]
        keyboard = ReplyKeyboardMarkup(indicators, one_time_keyboard=True)

        if update.callback_query:
            await update.callback_query.message.reply_text("Оберіть технічні індикатори:", reply_markup=keyboard)

    @staticmethod
    async def handle_indicator_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка вибору індикаторів та виконання аналізу"""
        selected_indicators = update.message.text if update.message else update.callback_query.data

        if selected_indicators == "Усі":
            context.user_data['selected_indicators'] = ["RSI", "MACD", "Bollinger Bands", "Stochastic"]
        else:
            context.user_data['selected_indicators'] = [selected_indicators]

        response_text = f"Індикатори: {', '.join(context.user_data['selected_indicators'])}"

        if update.message:
            await update.message.reply_text(response_text)
        elif update.callback_query:
            await update.callback_query.message.reply_text(response_text)

        await IndicatorHandler.show_info(update, context)

    @staticmethod
    async def show_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Відображення технічного аналізу за обраними індикаторами та відправка PNG картинки"""
        symbol = context.user_data.get('selected_pair')
        selected_indicators = context.user_data.get('selected_indicators', [])

        try:
            # Get historical data
            data = get_historical_data(symbol, "1d")
            logger.info(f"Fetched data: {data.head()}")  # Log the first few rows for debugging

            # Ensure required columns exist
            if 'timestamp' not in data.columns or 'close' not in data.columns:
                raise ValueError("Expected columns 'timestamp' and 'close' are missing in the data.")

            # Convert timestamps to dates
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

            # Create plot
            fig, ax = plt.subplots()
            ax.plot(data['date'], data['close'], label='Close Price')

            # Add indicators
            if "RSI" in selected_indicators:
                rsi = calculate_rsi(data)
                ax.plot(data['date'], rsi, label='RSI')

            if "MACD" in selected_indicators:
                macd_line, macd_signal = calculate_macd(data['close'])
                ax.plot(data['date'], macd_line, label='MACD Line')
                ax.plot(data['date'], macd_signal, label='MACD Signal')

            if "Bollinger Bands" in selected_indicators:
                upper_band, lower_band = calculate_bollinger_bands(data)
                ax.plot(data['date'], upper_band, label='Upper Band')
                ax.plot(data['date'], lower_band, label='Lower Band')

            if "Stochastic" in selected_indicators:
                stoch_k, stoch_d = calculate_stochastic(data)
                ax.plot(data['date'], stoch_k, label='Stochastic K')
                ax.plot(data['date'], stoch_d, label='Stochastic D')

            # Customize plot
            ax.set_title(f"{symbol} - Indicators")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()

            # Save plot to PNG
            png_image = io.BytesIO()
            plt.savefig(png_image, format='png')
            png_image.seek(0)  # Rewind the file pointer to the beginning of the file
            plt.close(fig)

            # Send PNG image
            if update.message:
                await update.message.reply_document(document=InputFile(png_image, filename='report.png'))
            elif update.callback_query:
                await update.callback_query.message.reply_document(document=InputFile(png_image, filename='report.png'))

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            if update.message:
                await update.message.reply_text('Виникла помилка при отриманні даних.')
            elif update.callback_query:
                await update.callback_query.message.reply_text('Виникла помилка при отриманні даних.')
