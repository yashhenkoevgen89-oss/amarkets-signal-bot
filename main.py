import os
import asyncio
from datetime import datetime, date

import pandas as pd
import numpy as np
import requests

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


# ======================
# CONFIG
# ======================

BOT_TOKEN = os.getenv("BOT_TOKEN")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

MODE = os.getenv("MODE", "SIGNAL")
AUTO_INTERVAL = int(os.getenv("AUTO_INTERVAL", "60"))

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


# ======================
# SETTINGS
# ======================

AUTO_ENABLED = False
CHAT_ID = None

SELECTED_PAIR = "EURUSD"
SELECTED_EXPIRATION = "5 мин"

sent_signals = set()
pair_cache = {}
signal_history = []

current_pair_index = 0


# ======================
# PAIRS
# ======================

PAIRS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP",
}

PAIR_LIST = list(PAIRS.keys())


# ======================
# EXPIRATIONS
# ======================

EXPIRATIONS = [
    "1 мин",
    "3 мин",
    "5 мин",
    "15 мин",
    "30 мин",
    "1 час",
]


# ======================
# STATISTICS
# ======================

stats = {
    "total": 0,
    "buy": 0,
    "sell": 0,
    "day": 0,
    "week": 0,
    "month": 0,
}


# ======================
# KEYBOARDS
# ======================

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="📊 Статус"),
            KeyboardButton(text="🔎 Сканер")
        ],
        [
            KeyboardButton(text="🏆 Лучшая"),
            KeyboardButton(text="🥇 Топ-3")
        ],
        [
            KeyboardButton(text="📡 Сигнал"),
            KeyboardButton(text="💱 Инструмент")
        ],
        [
            KeyboardButton(text="⏱ Экспирация"),
            KeyboardButton(text="⚙️ Настройки")
        ],
        [
            KeyboardButton(text="📈 Статистика"),
            KeyboardButton(text="📅 День")
        ],
        [
            KeyboardButton(text="🗓 Неделя"),
            KeyboardButton(text="📆 Месяц")
        ],
        [
            KeyboardButton(text="🟢 Авто ВКЛ"),
            KeyboardButton(text="🔴 Авто ВЫКЛ")
        ],
    ],
    resize_keyboard=True
)


pair_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="EURUSD"),
            KeyboardButton(text="GBPUSD")
        ],
        [
            KeyboardButton(text="USDJPY"),
            KeyboardButton(text="USDCHF")
        ],
        [
            KeyboardButton(text="USDCAD"),
            KeyboardButton(text="AUDUSD")
        ],
        [
            KeyboardButton(text="NZDUSD"),
            KeyboardButton(text="EURJPY")
        ],
        [
            KeyboardButton(text="GBPJPY"),
            KeyboardButton(text="EURGBP")
        ],
        [
            KeyboardButton(text="⬅️ Назад")
        ],
    ],
    resize_keyboard=True
)


expiration_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="⏱ 1 мин"),
            KeyboardButton(text="⏱ 3 мин")
        ],
        [
            KeyboardButton(text="⏱ 5 мин"),
            KeyboardButton(text="⏱ 15 мин")
        ],
        [
            KeyboardButton(text="⏱ 30 мин"),
            KeyboardButton(text="⏱ 1 час")
        ],
        [
            KeyboardButton(text="⬅️ Назад")
        ],
    ],
    resize_keyboard=True
)

# ======================
# HELPERS
# ======================

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return date.today().isoformat()


def get_interval():

    if SELECTED_EXPIRATION in ["1 мин", "3 мин"]:
        return "1min"

    if SELECTED_EXPIRATION in ["5 мин"]:
        return "5min"

    if SELECTED_EXPIRATION in ["15 мин"]:
        return "15min"

    if SELECTED_EXPIRATION in ["30 мин"]:
        return "30min"

    return "1h"


def clean_pair_name(symbol):
    return symbol.replace("/", "")


# ======================
# MARKET DATA
# ======================

def get_market_data(symbol, outputsize=300):

    try:
        api_symbol = PAIRS.get(symbol, symbol)

        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": api_symbol,
                "interval": get_interval(),
                "outputsize": outputsize,
                "apikey": TWELVE_DATA_API_KEY,
            },
            timeout=15
        )

        data = response.json()

        if "values" not in data:
            print(f"TwelveData error {symbol}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna()
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Market data error {symbol}: {e}")
        return pd.DataFrame()


# ======================
# INDICATORS
# ======================

def add_indicators(df):

    if df.empty:
        return df

    try:
        df["ema20"] = EMAIndicator(
            close=df["close"],
            window=20
        ).ema_indicator()

        df["ema50"] = EMAIndicator(
            close=df["close"],
            window=50
        ).ema_indicator()

        df["ema200"] = EMAIndicator(
            close=df["close"],
            window=200
        ).ema_indicator()

        df["rsi"] = RSIIndicator(
            close=df["close"],
            window=14
        ).rsi()

        macd = MACD(close=df["close"])

        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )

        df["adx"] = adx.adx()

        bb = BollingerBands(close=df["close"])

        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()

        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )

        df["stoch"] = stoch.stoch()

        return df

    except Exception as e:
        print(f"Indicator error: {e}")
        return df

# ======================
# SIGNAL ENGINE
# ======================

def analyze_pair(symbol):

    df = get_market_data(symbol)

    if df.empty:
        return None

    df = add_indicators(df)

    if len(df) < 210:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        close = float(last["close"])
        prev_close = float(prev["close"])

        rsi = float(last["rsi"])
        adx = float(last["adx"])

        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])

        macd = float(last["macd"])
        macd_signal = float(last["macd_signal"])

        stoch = float(last["stoch"])

    except Exception:
        return None

    buy_score = 0
    sell_score = 0

    if close > prev_close:
        buy_score += 10

    if close < prev_close:
        sell_score += 10

    if ema20 > ema50 > ema200:
        buy_score += 25

    if ema20 < ema50 < ema200:
        sell_score += 25

    if macd > macd_signal:
        buy_score += 20

    if macd < macd_signal:
        sell_score += 20

    if 45 <= rsi <= 70:
        buy_score += 15

    if 30 <= rsi <= 55:
        sell_score += 15

    if adx >= 18:
        buy_score += 15
        sell_score += 15

    if stoch < 80:
        buy_score += 10

    if stoch > 20:
        sell_score += 10

    direction = "HOLD"
    score = max(buy_score, sell_score)

    if buy_score > sell_score and buy_score >= 70:
        direction = "BUY"

    elif sell_score > buy_score and sell_score >= 70:
        direction = "SELL"

    else:
        return None

    return {
        "symbol": symbol,
        "signal": direction,
        "score": score,
        "price": round(close, 5),
        "rsi": round(rsi, 1),
        "adx": round(adx, 1),
        "ema": "BUY" if ema20 > ema50 else "SELL",
        "macd": "BUY" if macd > macd_signal else "SELL",
    }


def signal_rating(score):

    if score >= 90:
        return "🔥 STRONG"

    if score >= 80:
        return "✅ GOOD"

    return "⚪ NORMAL"


def build_signal_message(signal_data):

    if not signal_data:
        return "⚪ Сигнал не найден"

    emoji = "🟢" if signal_data["signal"] == "BUY" else "🔴"

    return (
        f"🚀 СИГНАЛ\n\n"
        f"Инструмент:\n{signal_data['symbol']}\n\n"
        f"Направление:\n{emoji} {signal_data['signal']}\n\n"
        f"Цена:\n{signal_data['price']}\n\n"
        f"Экспирация:\n{SELECTED_EXPIRATION}\n\n"
        f"Сила сигнала:\n{signal_data['score']}%\n\n"
        f"RSI: {signal_data['rsi']}\n"
        f"ADX: {signal_data['adx']}\n"
        f"EMA: {signal_data['ema']}\n"
        f"MACD: {signal_data['macd']}\n\n"
        f"{signal_rating(signal_data['score'])}"
    )

# ======================
# SCANNER
# ======================

def save_signal(signal_data):

    if not signal_data:
        return

    item = {
        "time": now(),
        "date": today_str(),
        "symbol": signal_data["symbol"],
        "signal": signal_data["signal"],
        "score": signal_data["score"],
        "price": signal_data["price"],
    }

    signal_history.append(item)

    stats["total"] += 1

    if signal_data["signal"] == "BUY":
        stats["buy"] += 1

    if signal_data["signal"] == "SELL":
        stats["sell"] += 1

    stats["day"] += 1
    stats["week"] += 1
    stats["month"] += 1


def scan_one(symbol):

    signal_data = analyze_pair(symbol)

    pair_cache[symbol] = {
        "time": now(),
        "signal": signal_data,
    }

    return signal_data


def scan_all():

    results = []

    for symbol in PAIR_LIST:

        signal_data = scan_one(symbol)

        if signal_data:
            results.append(signal_data)

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results


def get_best_signal():

    results = scan_all()

    if not results:
        return None

    return results[0]


def get_top3_signals():

    results = scan_all()

    return results[:3]


# ======================
# AUTO SCANNER
# ======================

async def auto_scanner():

    global CHAT_ID

    while True:

        if AUTO_ENABLED and CHAT_ID:

            try:

                signal_data = analyze_pair(SELECTED_PAIR)

                if signal_data:

                    signal_id = (
                        signal_data["symbol"]
                        + signal_data["signal"]
                        + str(signal_data["score"])
                    )

                    if signal_id not in sent_signals:

                        sent_signals.add(signal_id)

                        save_signal(signal_data)

                        await bot.send_message(
                            CHAT_ID,
                            "🚨 Новый сигнал\n\n"
                            + build_signal_message(signal_data)
                        )

            except Exception as e:

                print(f"Auto scanner error: {e}")

        await asyncio.sleep(AUTO_INTERVAL)

# ======================
# TELEGRAM FUNCTIONS
# ======================

async def show_status(message):

    mode = "🟢 ВКЛ" if AUTO_ENABLED else "🔴 ВЫКЛ"

    await message.answer(

        f"📊 Статус\n\n"

        f"Авто режим:\n"
        f"{mode}\n\n"

        f"Инструмент:\n"
        f"{SELECTED_PAIR}\n\n"

        f"Экспирация:\n"
        f"{SELECTED_EXPIRATION}",

        reply_markup=keyboard
    )


async def show_signal(message):

    signal_data = analyze_pair(
        SELECTED_PAIR
    )

    if signal_data:
        save_signal(signal_data)

    await message.answer(

        build_signal_message(
            signal_data
        ),

        reply_markup=keyboard
    )


async def show_scanner(message):

    results = scan_all()

    if not results:

        await message.answer(
            "⚪ Сигналы не найдены",
            reply_markup=keyboard
        )

        return

    text = "🔎 Сканер\n\n"

    for row in results[:10]:

        emoji = (
            "🟢"
            if row["signal"] == "BUY"
            else "🔴"
        )

        text += (
            f"{row['symbol']}\n"
            f"{emoji} {row['signal']}\n"
            f"{row['score']}%\n\n"
        )

    await message.answer(
        text,
        reply_markup=keyboard
    )


async def show_best(message):

    signal_data = get_best_signal()

    await message.answer(

        "🏆 Лучшая возможность\n\n"
        + build_signal_message(
            signal_data
        ),

        reply_markup=keyboard
    )


async def show_top3(message):

    top3 = get_top3_signals()

    if not top3:

        await message.answer(
            "⚪ Нет сигналов",
            reply_markup=keyboard
        )

        return

    text = "🥇 TOP-3\n\n"

    for i, row in enumerate(
        top3,
        start=1
    ):

        emoji = (
            "🟢"
            if row["signal"] == "BUY"
            else "🔴"
        )

        text += (
            f"{i}. {row['symbol']}\n"
            f"{emoji} {row['signal']}\n"
            f"{row['score']}%\n\n"
        )

    await message.answer(
        text,
        reply_markup=keyboard
    )


async def show_statistics(message):

    await message.answer(

        f"📈 Статистика\n\n"

        f"Всего сигналов:\n"
        f"{stats['total']}\n\n"

        f"BUY:\n"
        f"{stats['buy']}\n\n"

        f"SELL:\n"
        f"{stats['sell']}",

        reply_markup=keyboard
    )


async def show_day(message):

    await message.answer(

        f"📅 День\n\n"

        f"Сигналов:\n"
        f"{stats['day']}",

        reply_markup=keyboard
    )


async def show_week(message):

    await message.answer(

        f"🗓 Неделя\n\n"

        f"Сигналов:\n"
        f"{stats['week']}",

        reply_markup=keyboard
    )


async def show_month(message):

    await message.answer(

        f"📆 Месяц\n\n"

        f"Сигналов:\n"
        f"{stats['month']}",

        reply_markup=keyboard
    )

# ======================
# START
# ======================

@dp.message(Command("start"))
async def start_cmd(message: types.Message):

    global CHAT_ID

    CHAT_ID = message.chat.id

    await message.answer(
        "🚀 AMARKETS SIGNAL BOT STARTED",
        reply_markup=keyboard
    )


# ======================
# TEXT ROUTER
# ======================

@dp.message()
async def text_router(message: types.Message):

    global AUTO_ENABLED
    global SELECTED_PAIR
    global SELECTED_EXPIRATION
    global CHAT_ID

    CHAT_ID = message.chat.id

    text = message.text or ""

    # ---------- STATUS ----------

    if "📊" in text:
        await show_status(message)

    elif "📡" in text:
        await show_signal(message)

    elif "🔎" in text:
        await show_scanner(message)

    elif "🏆" in text:
        await show_best(message)

    elif "🥇" in text:
        await show_top3(message)

    elif "📈" in text:
        await show_statistics(message)

    elif "📅" in text:
        await show_day(message)

    elif "🗓" in text:
        await show_week(message)

    elif "📆" in text:
        await show_month(message)

    # ---------- PAIRS ----------

    elif "💱" in text:

        await message.answer(
            "Выберите валютную пару",
            reply_markup=pair_keyboard
        )

    elif text in PAIR_LIST:

        SELECTED_PAIR = text

        await message.answer(
            f"✅ Инструмент выбран\n\n{SELECTED_PAIR}",
            reply_markup=keyboard
        )

    # ---------- EXPIRATION ----------

    elif "⏱ Экспирация" in text:

        await message.answer(
            "Выберите экспирацию",
            reply_markup=expiration_keyboard
        )

    elif text.startswith("⏱"):

        SELECTED_EXPIRATION = (
            text
            .replace("⏱", "")
            .strip()
        )

        await message.answer(
            f"✅ Экспирация:\n\n{SELECTED_EXPIRATION}",
            reply_markup=keyboard
        )

    # ---------- AUTO ----------

    elif "🟢" in text:

        AUTO_ENABLED = True

        await message.answer(
            "🟢 Авто режим включен",
            reply_markup=keyboard
        )

    elif "🔴" in text:

        AUTO_ENABLED = False

        await message.answer(
            "🔴 Авто режим выключен",
            reply_markup=keyboard
        )

    # ---------- BACK ----------

    elif "⬅️" in text:

        await message.answer(
            "Главное меню",
            reply_markup=keyboard
        )

    else:

        await message.answer(
            "❓ Команда не распознана",
            reply_markup=keyboard
        )


# ======================
# MAIN
# ======================

async def main():

    asyncio.create_task(
        auto_scanner()
    )

    print(
        "AMARKETS SIGNAL BOT STARTED"
    )

    await dp.start_polling(
        bot
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
