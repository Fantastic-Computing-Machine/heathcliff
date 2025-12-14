# ABOUTME: Heathcliff's personality-driven greeting generator
# ABOUTME: Creates time-aware, weather-aware greetings with British butler charm

import random
from datetime import datetime
from typing import Optional

from config import get_config
from logger import logger


def get_time_of_day() -> str:
    """Get time of day category for greeting."""
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_weather_description(location: Optional[str] = None) -> Optional[str]:
    """
    Fetch current weather briefly for greeting context.

    Args:
        location: City name (uses config default if None)

    Returns:
        Brief weather description or None if unavailable
    """
    try:
        import requests

        config = get_config()
        api_key = config.openweathermap_key

        if not api_key:
            return None

        if location is None:
            location = config.get("weather.default_city", "London")

        units = config.get("weather.units", "metric")
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": api_key, "units": units}

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = int(data["main"]["temp"])
        description = data["weather"][0]["main"].lower()  # "Clear", "Rain", etc.
        temp_unit = "°C" if units == "metric" else "°F"

        return f"{description}, {temp}{temp_unit}"

    except Exception as e:
        logger.debug(f"Could not fetch weather for greeting: {e}")
        return None


def generate_greeting(user_name: str = "Sir", include_weather: bool = True) -> str:
    """
    Generate a sophisticated British butler greeting based on time and weather.

    Args:
        user_name: Name to address user (default: "Sir")
        include_weather: Whether to include weather context

    Returns:
        Personalised Heathcliff greeting
    """
    time_period = get_time_of_day()

    # Base greetings by time of day
    greetings = {
        "morning": [
            f"Good morning, {user_name}. I trust you slept well?",
            f"Good morning, {user_name}. Shall we make today productive?",
            f"Rise and shine, {user_name}. How may I assist you this fine morning?",
            f"Good morning, {user_name}. Coffee first, or shall we dive straight in?",
        ],
        "afternoon": [
            f"Good afternoon, {user_name}. How may I be of service?",
            f"Good afternoon, {user_name}. I trust the day is treating you well?",
            f"Afternoon, {user_name}. What's on the agenda?",
            f"Good afternoon, {user_name}. Shall we tackle that to-do list?",
        ],
        "evening": [
            f"Good evening, {user_name}. How was your day?",
            f"Good evening, {user_name}. Time to unwind, or still working?",
            f"Evening, {user_name}. What can I help you with tonight?",
            f"Good evening, {user_name}. I trust it's been a productive day?",
        ],
        "night": [
            f"Burning the midnight oil, {user_name}?",
            f"Rather late, isn't it, {user_name}? How may I assist?",
            f"Good evening, {user_name}. Still awake, I see.",
            f"Late night session, {user_name}? I'm here to help.",
        ],
    }

    base_greeting = random.choice(greetings[time_period])

    # Add weather context if available
    if include_weather:
        weather = get_weather_description()
        if weather:
            weather_comments = {
                "clear": [
                    f"Lovely weather outside - {weather}.",
                    f"Rather pleasant out there - {weather}.",
                    f"Splendid conditions - {weather}.",
                ],
                "clouds": [
                    f"Bit overcast today - {weather}.",
                    f"Grey skies, I'm afraid - {weather}.",
                    f"Cloudy conditions - {weather}.",
                ],
                "rain": [
                    f"Do bring an umbrella - it's {weather}.",
                    f"Rather wet outside - {weather}.",
                    f"Raining, I'm afraid - {weather}.",
                ],
                "snow": [
                    f"Winter has arrived - {weather}.",
                    f"Wrap up warm - {weather}.",
                    f"Rather chilly - {weather}.",
                ],
                "thunderstorm": [
                    f"Bit dramatic outside - {weather}.",
                    f"Thor's having a tantrum - {weather}.",
                    f"Stormy weather - {weather}.",
                ],
            }

            # Match weather description to comment
            for condition, comments in weather_comments.items():
                if condition in weather.lower():
                    weather_note = random.choice(comments)
                    base_greeting = f"{base_greeting} {weather_note}"
                    break
            else:
                # Generic weather mention
                base_greeting = f"{base_greeting} It's {weather} outside."

    return base_greeting


def generate_return_greeting(
    user_name: str = "Sir", hours_since_last: Optional[float] = None
) -> str:
    """
    Generate greeting for returning users based on absence duration.

    Args:
        user_name: Name to address user
        hours_since_last: Hours since last interaction

    Returns:
        Appropriate return greeting
    """
    if hours_since_last is None or hours_since_last < 0.5:
        # Continuous conversation - no greeting needed
        return ""

    if hours_since_last < 4:
        # Short absence
        return random.choice(
            [
                f"Welcome back, {user_name}.",
                f"Ah, there you are, {user_name}.",
                f"Back so soon, {user_name}?",
            ]
        )
    elif hours_since_last < 12:
        # Several hours
        return random.choice(
            [
                f"Welcome back, {user_name}. I trust you've been well?",
                f"Good to see you again, {user_name}. How may I assist?",
                f"Ah, {user_name}. How did things go?",
            ]
        )
    else:
        # Long absence (more than 12 hours)
        time_period = get_time_of_day()
        if time_period == "morning":
            return random.choice(
                [
                    f"Good morning, {user_name}. Pleasant dreams, I hope?",
                    f"Good morning, {user_name}. Ready for a fresh start?",
                ]
            )
        else:
            return random.choice(
                [
                    f"Welcome back, {user_name}. I trust your day has been eventful?",
                    f"Good to see you, {user_name}. Shall we catch up?",
                    f"Ah, {user_name}. Long time, no see. How have you been?",
                ]
            )
