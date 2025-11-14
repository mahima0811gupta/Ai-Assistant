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
    """Extract city name for weather query."""
    return query.strip()


# TOOL 1 — Play YouTube Song


class YouTubeMusicInput(BaseModel):
    song_name: str = Field(description="Song name to play on YouTube")


class PlayYouTubeSongTool(BaseTool):
    name: str = "play_youtube_song"
    description: str = "Play a song on YouTube using Chrome."
    args_schema: Type[BaseModel] = YouTubeMusicInput

    def _run(self, song_name: str):
        try:
            url = pywhatkit.playonyt(song_name, open_video=False)
            chrome_path = r"C:/Program Files/Google/Chrome/Application/chrome.exe %s"
            webbrowser.get(chrome_path).open(url)
            return f"Playing '{song_name}'."
        except Exception as e:
            return f"Error playing song: {e}"


# TOOL 2 — Weather


class WeatherInput(BaseModel):
    city: str = Field(description="City name to fetch weather for")


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Fetch current weather for any city."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        city_name = extract_city(city)

        if not api_key:
            return "OpenWeather API key missing."

        try:
            res = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city_name, "appid": api_key, "units": "metric"}
            ).json()

            if "main" not in res:
                return f"City '{city_name}' not found."

            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]

            return f"Weather in {city_name}: {temp}°C, {desc}"
        except Exception as e:
            return f"Weather error: {e}"

# TOOL 3 — Web Search (Tavily)

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")


class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet using Tavily."
    args_schema: Type[BaseModel] = WebSearchInput

    engine: ClassVar = TavilySearch(max_results=3)

    def _run(self, query: str):
        try:
            cleaned = clean_query(query)
            results = self.engine.run(cleaned)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Web search failed: {e}"


# TOOL 4 — Open Application


class AppInput(BaseModel):
    app_name: str = Field(description="Application to open (chrome/notepad)")


class OpenApplicationTool(BaseTool):
    name: str = "open_application"
    description: str = "Open applications like Chrome, Notepad, Calculator, Explorer."
    args_schema: Type[BaseModel] = AppInput

    known_apps: ClassVar[dict] = {
        "notepad": "notepad.exe",
        "chrome": "chrome.exe",
        "calculator": "calc.exe",
        "explorer": "explorer.exe"
    }

    def _run(self, app_name: str):
        name = app_name.lower()

        if name not in self.known_apps:
            return "Unknown app. Try: notepad, chrome, calculator, explorer."

        try:
            subprocess.Popen(["start", self.known_apps[name]], shell=True)
            return f"Opened {name}."
        except Exception as e:
            return f"Failed to open {name}: {e}"


#
# TOOL 5 — Write Note

class NoteInput(BaseModel):
    note_content: str
    append: bool = False


class WriteNoteTool(BaseTool):
    name: str = "write_note"
    description: str = "Write or append text into a note file."
    args_schema: Type[BaseModel] = NoteInput

    def _run(self, note_content: str, append: bool = False):
        path = "dora_note.txt"
        mode = "a" if append else "w"

        try:
            with open(path, mode, encoding="utf-8") as f:
                f.write(note_content + "\n")

            subprocess.Popen(["notepad", path])
            return "Note saved."
        except Exception as e:
            return f"Note writing error: {e}"


# TOOL 6 — Memory Save


class RememberInput(BaseModel):
    fact_to_remember: str


class RememberThisTool(BaseTool):
    name: str = "remember_this"
    description: str = "Store user information in memory."
    args_schema: Type[BaseModel] = RememberInput

    def _run(self, fact_to_remember: str):
        return save_memory(fact_to_remember)

# TOOL 7 — Memory Recall


class RecallInput(BaseModel):
    query: str


class RecallInfoTool(BaseTool):
    name: str = "recall_information"
    description: str = "Retrieve previously stored memories."
    args_schema: Type[BaseModel] = RecallInput

    def _run(self, query: str):
        memories = recall_memories(query)
        if not memories:
            return "No matching memories."
        return "\n".join(memories)



# Export all tools (for ai_agent)

play_youtube_song = PlayYouTubeSongTool()
open_application = OpenApplicationTool()
get_weather = GetWeatherTool()
tavily_web_search = TavilySearchTool()
write_note = WriteNoteTool()
remember_this = RememberThisTool()
recall_information = RecallInfoTool()
