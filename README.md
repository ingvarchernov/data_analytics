Welcome to the New Project repository! This project demonstrates a set of skills and technologies designed to provide valuable functionality and insights, ranging from data processing and analysis to web interface implementation. Below is an overview of the project's components and technical aspects.

Project Overview
The project is structured to ensure flexibility, efficiency, and functionality, making it suitable for both individual usage and as part of a larger system. The primary objectives and features include:

Data Extraction: Efficiently gathers financial data using APIs.
Technical Indicators: Incorporates popular indicators like RSI, MACD, Bollinger Bands, and Stochastic Oscillator.
Machine Learning: Uses neural networks and deep learning architectures for predictions.
User Interface: Integrates real-time data visualization and interaction.
Key Features
1. Data Extraction
The project retrieves data from financial sources using secure and optimized methods, allowing for smooth data processing.

2. Technical Indicators
The indicators used in this project are essential for financial data analysis, providing insightful information for better decision-making.

3. Machine Learning
Implemented models include LSTM and Bidirectional LSTM with attention mechanisms, providing accurate predictions based on technical indicators.

4. Web Interface
The project includes a Vue.js and FastAPI setup for a real-time user interface, presenting technical indicators and predictions to the user.

5. Telegram Bot Integration
Control and monitoring functionalities are available via a Telegram bot, allowing users to stay informed without needing to access the main web application.

Tech Stack
Backend: Python, FastAPI
Frontend: Vue.js
Machine Learning: Keras, TensorFlow
Data Storage: PyMySQL
Messaging: Telegram Bot API
Visualization: matplotlib, mplfinance
Getting Started
Prerequisites
Python 3.8+
Node.js and npm
MySQL or compatible database
Installation
Clone the repository:

bash
Копіювати код
git clone https://github.com/ingvarchernov/new.git
cd new
Set up a virtual environment:

bash
Копіювати код
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
Install the necessary Python packages:

bash
Копіювати код
pip install -r requirements.txt
Set up the database and environment variables (instructions provided in config/README.md).

Running the Application
Backend: Start FastAPI:

bash
Копіювати код
uvicorn main:app --reload
Frontend: Navigate to the frontend folder and start Vue.js:

bash
Копіювати код
cd frontend
npm install
npm run serve
Telegram Bot: Configure and run the bot (see bot/README.md).

Contributing
Contributions to this project are welcome. Please submit a pull request or contact the repository owner for major changes.

License
This project is licensed under the MIT License.

