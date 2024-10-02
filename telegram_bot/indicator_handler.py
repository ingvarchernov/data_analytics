import pandas as pd
<<<<<<< HEAD
import io
import matplotlib.pyplot as plt
import logging
import mplfinance as mpf
from io import BytesIO
from telegram import Update, ReplyKeyboardMarkup, InputFile
from telegram.ext import ContextTypes
from bi_api.data_extraction import get_historical_data
from indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic

logger = logging.getLogger(__name__)

=======
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


>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211
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
<<<<<<< HEAD
        """Відображення свічкового графіка з технічними індикаторами та відправка PNG картинки"""
=======
        """Відображення технічного аналізу за обраними індикаторами та відправка PNG картинки"""
>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211
        symbol = context.user_data.get('selected_pair')
        selected_indicators = context.user_data.get('selected_indicators', [])

        try:
<<<<<<< HEAD
            # Отримуємо історичні дані
            data = get_historical_data(symbol, "1d")
            logger.info(f"Fetched data: {data.head()}")

            # Перевірка наявності необхідних колонок
            if 'timestamp' not in data.columns or 'close' not in data.columns:
                raise ValueError("Очікувані колонки 'timestamp' і 'close' відсутні в даних.")

            # Конвертуємо timestamp в формат дати
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('date', inplace=True)

            # Конвертуємо дані у числовий формат
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')  # Конвертуємо в числовий формат з обробкою помилок

            # Видаляємо рядки з пропущеними значеннями, якщо такі є
            data.dropna(subset=numeric_columns, inplace=True)

            # Створення свічкового графіка
            ohlc_data = data[['open', 'high', 'low', 'close', 'volume']]

            # Налаштування індикаторів
            add_plots = []
            if "RSI" in selected_indicators:
                rsi = calculate_rsi(data)
                add_plots.append(mpf.make_addplot(rsi, panel=1, color='blue', ylabel='RSI'))

            if "MACD" in selected_indicators:
                macd_line, macd_signal = calculate_macd(data['close'])
                add_plots.append(mpf.make_addplot(macd_line, panel=2, color='orange', ylabel='MACD Line'))
                add_plots.append(mpf.make_addplot(macd_signal, panel=2, color='red', ylabel='MACD Signal'))

            if "Bollinger Bands" in selected_indicators:
                upper_band, lower_band = calculate_bollinger_bands(data)
                add_plots.append(mpf.make_addplot(upper_band, color='green'))
                add_plots.append(mpf.make_addplot(lower_band, color='purple'))

            if "Stochastic" in selected_indicators:
                stoch_k, stoch_d = calculate_stochastic(data)
                add_plots.append(mpf.make_addplot(stoch_k, panel=3, color='cyan', ylabel='Stochastic K'))
                add_plots.append(mpf.make_addplot(stoch_d, panel=3, color='magenta', ylabel='Stochastic D'))

            # Налаштування стилю графіка
            style = mpf.make_mpf_style(base_mpf_style='charles', facecolor='white')

            # Створення графіка
            fig, axlist = mpf.plot(
                ohlc_data,
                type='candle',  # Свічковий графік
                volume=True,    # Додаємо об'єм
                addplot=add_plots,  # Додаємо технічні індикатори
                style=style,    # Стиль графіка
                returnfig=True,
                figsize=(60, 30)  # Розмір графіка
            )

            logger.info(f"Fig with fig_size: {fig}")
            # Збереження графіка в буфер
            image_stream = BytesIO()
            fig.savefig(image_stream, format='png')
            image_stream.seek(0)  # Переміщаємо курсор на початок
            plt.close(fig)

            # Відправка графіка в Telegram
            await update.message.reply_photo(photo=InputFile(image_stream, filename='chart.png'))

        except Exception as e:
            logger.error(f"Помилка при отриманні даних: {e}")
            await update.message.reply_text('Виникла помилка при отриманні даних.')
=======
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
>>>>>>> 9ee9d2292d3411fcadabb0a0fde575f6cb563211
