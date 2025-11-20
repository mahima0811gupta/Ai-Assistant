import os
import subprocess
import webbrowser
import pywhatkit
import requests
from dotenv import load_dotenv
from typing import ClassVar, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
import re
import json

from memory_manager import save_memory, recall_memories

load_dotenv()

# Helper Functions

def clean_query(query: str) -> str:
    """Clean unnecessary noise or extra spaces."""
    return re.sub(r"\s+", " ", query).strip()


def extract_city(query: str) -> str:
    """Extract city name from weather queries with improved parsing"""
    query_lower = query.lower().strip()
    print(f"🔍 Extracting city from: '{query_lower}'")
    
    # Step 1: Try regex pattern matching first
    patterns = [
        r'what\s+is\s*,?\s*what\s+is\s+the\s+(?:weather|temperature|temp)\s+(?:of|in|at|for)\s+([a-z\s]+)',
        r'what\s+is\s+the\s+(?:weather|temperature|temp)\s+(?:of|in|at|for)\s+([a-z\s]+)',
        r'what\'s\s+the\s+(?:weather|temperature|temp)\s+(?:of|in|at|for)\s+([a-z\s]+)',
        r'(?:weather|temperature|temp)\s+(?:of|in|at|for)\s+([a-z\s]+)',
        r'how\s+is\s+the\s+weather\s+(?:of|in|at)\s+([a-z\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            city = match.group(1).strip().replace('?', '')
            print(f"✅ Pattern matched, extracted: '{city}'")
            return city.title()
    
    # Step 2: Try to find known cities in the query
    known_cities = [
        'mumbai', 'delhi', 'lucknow', 'bangalore', 'chennai', 'kolkata', 
        'pune', 'hyderabad', 'ahmedabad', 'jaipur', 'surat', 'kanpur',
        'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
        'agra', 'varanasi', 'meerut', 'nashik', 'faridabad', 'ghaziabad',
        'new york', 'london', 'paris', 'tokyo', 'dubai', 'singapore'
    ]
    
    for city in known_cities:
        if city in query_lower:
            print(f"✅ Known city found: '{city}'")
            return city.title()
    
    # Step 3: Word filtering approach as fallback
    print(f"⚠️ Pattern matching failed, using word filtering")
    words = query_lower.replace('?', '').replace(',', '').split()
    filter_words = [
        'what', 'is', 'the', 'weather', 'temperature', 'temp', 'temps',
        'in', 'at', 'of', 'for', 'forecast', 'today', 'now', 'current',
        'tell', 'me', 'about', 'how', 'whats'
    ]
    
    city_words = [w for w in words if w not in filter_words and len(w) > 2]
    
    if city_words:
        city = ' '.join(city_words)
        print(f"✅ Extracted via filtering: '{city}'")
        return city.title()
    
    # Step 4: Default fallback
    print(f"⚠️ Could not extract city, using default")
    return "Mumbai"
# TOOL 1 — Play YouTube Song

class YouTubeMusicInput(BaseModel):
    song_name: str = Field(description="Song name to play on YouTube")


class PlayYouTubeSongTool(BaseTool):
    name: str = "play_youtube_song"
    description: str = "Play a song on YouTube using Chrome. Use this when user asks to play music or a specific song."
    args_schema: Type[BaseModel] = YouTubeMusicInput

    def _run(self, song_name: str):
        try:
            print(f"🎵 Attempting to play: {song_name}")
            url = pywhatkit.playonyt(song_name, open_video=False)
            chrome_path = r"C:/Program Files/Google/Chrome/Application/chrome.exe %s"
            webbrowser.get(chrome_path).open(url)
            return f"Now playing '{song_name}' on YouTube."
        except Exception as e:
            print(f"❌ YouTube error: {e}")
            return f"Sorry, I couldn't play '{song_name}'. Error: {e}"


# TOOL 2 — Enhanced Weather

class WeatherInput(BaseModel):
    city: str = Field(description="City name to fetch weather for")


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Fetch current weather information for any city including temperature, feels like, humidity, and conditions."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        # Use the standalone extract_city function
        city_name = extract_city(city)

        if not api_key:
            return "OpenWeather API key is missing. Please add it to your .env file."

        if not city_name or len(city_name) < 2:
            return "Please specify a valid city name."

        try:
            print(f"🌤️ Fetching weather for: {city_name}")
            res = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city_name, "appid": api_key, "units": "metric"},
                timeout=10
            ).json()

            if "main" not in res:
                return f"Sorry, I couldn't find weather information for '{city_name}'. Please check the city name."

            temp = res["main"]["temp"]
            feels_like = res["main"]["feels_like"]
            desc = res["weather"][0]["description"]
            humidity = res["main"]["humidity"]
            wind_speed = res.get("wind", {}).get("speed", 0)

            return (f"Weather in {city_name}: {temp}°C (feels like {feels_like}°C). "
                   f"{desc.capitalize()} with {humidity}% humidity and wind speed {wind_speed} m/s.")
        
        except requests.Timeout:
            return "Weather request timed out. Please try again."
        except Exception as e:
            print(f"❌ Weather error: {e}")
            return f"Sorry, I encountered an error fetching weather: {e}"


# TOOL 3 — Web Search (Tavily)

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for finding information on the internet")


class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet using Tavily for latest news, information, and answers to questions."
    args_schema: Type[BaseModel] = WebSearchInput

    engine: ClassVar = TavilySearch(max_results=5)

    def _run(self, query: str):
        try:
            cleaned = clean_query(query)
            print(f"🔍 Searching web for: {cleaned}")
            results = self.engine.run(cleaned)
            return json.dumps(results, indent=2)
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return f"Web search failed: {e}"


# TOOL 4 — Open Application

class AppInput(BaseModel):
    app_name: str = Field(description="Application name to open: chrome, notepad, calculator, explorer, or any installed app")


class OpenApplicationTool(BaseTool):
    name: str = "open_application"
    description: str = "Open system applications like Chrome, Notepad, Calculator, File Explorer, or any installed Windows application."
    args_schema: Type[BaseModel] = AppInput

    known_apps: ClassVar[dict] = {
        "notepad": "notepad.exe",
        "chrome": "chrome.exe",
        "calculator": "calc.exe",
        "explorer": "explorer.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "terminal": "cmd.exe"
    }

    def _run(self, app_name: str):
        name = app_name.lower()

        if name not in self.known_apps:
            return f"I don't know how to open '{app_name}'. I can open: {', '.join(self.known_apps.keys())}."

        try:
            print(f"🚀 Opening: {name}")
            subprocess.Popen(["start", self.known_apps[name]], shell=True)
            return f"Successfully opened {name}."
        except Exception as e:
            print(f"❌ App open error: {e}")
            return f"Failed to open {name}: {e}"


# TOOL 5 — Write Note


class NoteInput(BaseModel):
    note_content: str = Field(description="The content of the note.")
    append: bool = Field(description="Whether to append to existing note.", default=False)

class WriteNoteTool(BaseTool):
    name: str = "write_note"
    description: str = "Write or append to a note file."
    args_schema: Type[NoteInput] = NoteInput
    
    def _run(self, note_content: str, append: bool = False) -> str:
        file_path = "dora_note.txt"
        mode = "a" if append else "w"
        try:
            content_to_write = f"\n{note_content}" if append and os.path.exists(file_path) else note_content
            with open(file_path, mode) as f: 
                f.write(content_to_write)
            subprocess.Popen(["notepad.exe", file_path])
            action = "added to" if append else "created"
            return f"I've {action} your note and opened it in Notepad."
        except Exception as e:
            return f"Sorry, an error occurred while writing the note: {e}"


# TOOL 6 — Memory Save

class RememberInput(BaseModel):
    fact_to_remember: str = Field(description="Information or fact to remember about the user")


class RememberThisTool(BaseTool):
    name: str = "remember_this"
    description: str = "Store important user information in long-term memory for future reference."
    args_schema: Type[BaseModel] = RememberInput

    def _run(self, fact_to_remember: str):
        print(f"💾 Saving to memory: {fact_to_remember}")
        return save_memory(fact_to_remember)


# TOOL 7 — Memory Recall

class RecallInput(BaseModel):
    query: str = Field(description="Query to search for in stored memories")


class RecallInfoTool(BaseTool):
    name: str = "recall_information"
    description: str = "Retrieve previously stored memories and information about the user."
    args_schema: Type[BaseModel] = RecallInput

    def _run(self, query: str):
        print(f"🔍 Recalling memories for: {query}")
        memories = recall_memories(query)
        if not memories:
            return "I don't have any stored memories matching that query."
        return "Here's what I remember:\n" + "\n".join(memories)


# TOOL 8 — System Time

class GetTimeInput(BaseModel):
    format: str = Field(default="12h", description="Time format: '12h' or '24h'")


class GetTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Get the current system time."
    args_schema: Type[BaseModel] = GetTimeInput

    def _run(self, format: str = "12h"):
        from datetime import datetime
        try:
            now = datetime.now()
            if format == "24h":
                time_str = now.strftime("%H:%M:%S")
            else:
                time_str = now.strftime("%I:%M:%S %p")
            
            date_str = now.strftime("%A, %B %d, %Y")
            return f"Current time: {time_str} on {date_str}"
        except Exception as e:
            return f"Error getting time: {e}"


# Export all tools
play_youtube_song = PlayYouTubeSongTool()
open_application = OpenApplicationTool()
get_weather = GetWeatherTool()
tavily_web_search = TavilySearchTool()
write_note = WriteNoteTool()
remember_this = RememberThisTool()
recall_information = RecallInfoTool()
get_current_time = GetTimeTool()