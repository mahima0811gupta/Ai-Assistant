import cv2
import base64
from dotenv import load_dotenv
import os
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, ClassVar
from PIL import Image
import io
import webbrowser
import pywhatkit
import google.generativeai as genai
import pytesseract
from pyzbar.pyzbar import decode
import numpy as np
import subprocess
import requests
from langchain_tavily import TavilySearch
import re
import json

from memory_manager import save_memory, recall_memories

load_dotenv()

# --- Configurations & Initializations ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'
except Exception:
    print("Tesseract OCR not found. ReadTextTool may not work.")

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # FIXED: Use the correct model name - these are the valid options:
    # 'gemini-1.5-flash' (faster, cheaper)
    # 'gemini-1.5-pro' (more capable)
    # 'gemini-pro-vision' (older model, still works)
    gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini Vision model initialized successfully with gemini-1.5-flash")
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")
    gemini_vision_model = None

from shared_frame import get_current_frame

# --- Helper Functions ---
def capture_image() -> str:
    """Fallback image capture if shared frame is not available"""
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(5): 
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if ret:
                ret, buf = cv2.imencode('.jpg', frame)
                if ret: 
                    return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcam.")

def frame_to_pil(frame) -> Image:
    """Convert numpy frame to PIL Image"""
    if frame is None: 
        return None
    try:
        # Ensure the frame is in the correct format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Frame is already RGB
            return Image.fromarray(frame)
        else:
            print(f"⚠️ Unexpected frame shape: {frame.shape}")
            return None
    except Exception as e:
        print(f"❌ Error converting frame to PIL: {e}")
        return None

def extract_city_from_query(query: str) -> str:
    """Extract city name from weather queries"""
    cities = [
        'mumbai', 'delhi', 'lucknow', 'bangalore', 'chennai', 'kolkata', 
        'pune', 'hyderabad', 'ahmedabad', 'jaipur', 'surat', 'kanpur',
        'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri'
    ]
    query_lower = query.lower()
    for city in cities:
        if city in query_lower:
            return city.title()
    return query if len(query.split()) <= 2 else "Mumbai"  # Default fallback

def clean_query(query: str) -> str:
    """Clean query of unwanted sounds and artifacts"""
    import re
    
    # Remove common audio artifacts from transcription
    artifacts = [
        "[Sound of static/processing]",
        "[Sound of a phone ringing]",
        "[Sound of a person sighing or breathing heavily]",
        "[Sound of indistinct speech or possibly a beep]",
        "[Static]",
        "[Processing]",
        "[Background noise]",
        "[Breathing]",
        "[Sighing]",
        "[Beep]",
        "[Indistinct speech]",
        "[Heavy breathing]",
        "[Sound of breathing]",
        "[Sound of sighing]",
        "[Sound of beeping]",
        "[Noise]",
        "[Electronic tone]"
    ]
    
    cleaned = query
    
    # Remove exact matches first
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    
    # Remove any remaining patterns that look like audio artifacts
    patterns = [
        r'\[Sound of[^\]]*\]',
        r'\[.*?breathing.*?\]',
        r'\[.*?sighing.*?\]',
        r'\[.*?beep.*?\]',
        r'\[.*?noise.*?\]',
        r'\[.*?static.*?\]',
        r'\[.*?processing.*?\]',
        r'\[.*?indistinct.*?\]',
        r'\[.*?tone.*?\]',
    ]
    
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace and multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def format_search_results_safely(raw_results) -> str:
    """Safely format search results to prevent Gradio crashes"""
    try:
        if isinstance(raw_results, str):
            return raw_results
        
        if isinstance(raw_results, dict):
            if 'results' in raw_results:
                results = raw_results['results']
                formatted_results = []
                
                for i, result in enumerate(results[:3]):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', '')
                    
                    content = content[:200] + "..." if len(content) > 200 else content
                    
                    formatted_result = f"{i+1}. {title}\n   {content}"
                    if url and not url.startswith('javascript:'):
                        formatted_result += f"\n   Source: {url}"
                    
                    formatted_results.append(formatted_result)
                
                return "\n\n".join(formatted_results)
            else:
                return str(raw_results)[:500] + "..." if len(str(raw_results)) > 500 else str(raw_results)
        
        return str(raw_results)[:500] + "..." if len(str(raw_results)) > 500 else str(raw_results)
        
    except Exception as e:
        return f"Search completed but results formatting failed: {str(e)}"

# --- Tool Classes ---

class ImageAnalysisInput(BaseModel):
    query: str = Field(description="The user's question about the image.")

class AnalyzeImageWithQueryTool(BaseTool):
    name: str = "analyze_image_with_query"
    description: str = """Analyze the current webcam image with a specific query. 
    Use this for ANY visual question like:
    - "What am I wearing?"
    - "Am I wearing glasses?"
    - "How many people are in the frame?"
    - "What do you see?"
    - "Describe what's visible"
    """
    args_schema: Type[ImageAnalysisInput] = ImageAnalysisInput
    
    def _run(self, query: str) -> str:
        """
        Enhanced image analysis with better error handling and debugging
        """
        print(f"🔍 Image analysis requested with query: '{query}'")
        
        # Check if Gemini model is available
        if not gemini_vision_model:
            error_msg = "Error: Gemini Vision model is not initialized. Check your GOOGLE_API_KEY in .env file."
            print(f"❌ {error_msg}")
            return error_msg
        
        pil_image = None
        
        try:
            # Step 1: Try to get frame from shared state
            print("📸 Attempting to get frame from shared state...")
            current_frame = get_current_frame()
            
            if current_frame is not None:
                print(f"✅ Frame retrieved from shared state. Shape: {current_frame.shape}")
                pil_image = frame_to_pil(current_frame)
                
                if pil_image:
                    print(f"✅ Successfully converted to PIL Image. Size: {pil_image.size}")
                else:
                    print("⚠️ Failed to convert frame to PIL Image")
            else:
                print("⚠️ No frame available in shared state")
            
            # Step 2: Fallback to direct capture if needed
            if not pil_image:
                print("📸 Falling back to direct camera capture...")
                try:
                    b64_string = capture_image()
                    image_bytes = base64.b64decode(b64_string)
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    print(f"✅ Captured new frame via fallback. Size: {pil_image.size}")
                except Exception as capture_error:
                    print(f"❌ Fallback capture failed: {capture_error}")
                    return f"Error: Could not capture any image from the webcam. Details: {capture_error}"
            
            # Step 3: Verify we have a valid image
            if not pil_image:
                error_msg = "Error: Could not obtain a valid image from either shared state or direct capture."
                print(f"❌ {error_msg}")
                return error_msg
            
            # Step 4: Send to Gemini for analysis
            print(f"🤖 Sending image to Gemini with query: '{query}'")
            
            try:
                response = gemini_vision_model.generate_content([query, pil_image])
                
                # Check if response is valid
                if not response or not response.text:
                    print("⚠️ Gemini returned empty response")
                    return "I analyzed the image but couldn't generate a response. Please try again."
                
                print(f"✅ Gemini response received: {response.text[:100]}...")
                return response.text
                
            except Exception as gemini_error:
                print(f"❌ Gemini API error: {gemini_error}")
                return f"Error communicating with Gemini API: {gemini_error}"
            
        except Exception as e:
            error_msg = f"Sorry, an unexpected error occurred during image analysis: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
            return error_msg

class YouTubeMusicInput(BaseModel):
    song_name: str = Field(description="The name of the song and artist to play on YouTube.")

class PlayYouTubeSongTool(BaseTool):
    name: str = "play_youtube_song"
    description: str = "Use this to play a song on YouTube."
    args_schema: Type[YouTubeMusicInput] = YouTubeMusicInput
    
    def _run(self, song_name: str) -> str:
        try:
            print(f"Searching for '{song_name}' on YouTube...")
            video_url = pywhatkit.playonyt(song_name, open_video=False)
            if not video_url: 
                return f"Sorry, I could not find a video for '{song_name}'."
            chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
            webbrowser.get(chrome_path).open(video_url, new=2)
            return f"Now playing '{song_name}' for you in Chrome."
        except Exception as e:
            return "Sorry, I encountered an error while trying to play the song."

class ReadTextInput(BaseModel):
    pass

class ReadTextTool(BaseTool):
    name: str = "read_text_from_image"
    description: str = "Use this tool to read and extract any text visible in the current webcam view."
    args_schema: Type[ReadTextInput] = ReadTextInput
    
    def _run(self) -> str:
        try:
            current_frame = get_current_frame()
            if current_frame is None: 
                return "Error: Could not get an image from the webcam."
            pil_image = Image.fromarray(current_frame)
            text = pytesseract.image_to_string(pil_image)
            if not text.strip(): 
                return "I looked, but I could not find any text to read."
            return f"The text I see is: '{text.strip()}'"
        except Exception as e:
            return f"Sorry, I encountered an error while trying to read the text: {e}"

class ScanCodeInput(BaseModel):
    pass

class ScanCodeTool(BaseTool):
    name: str = "scan_qr_or_barcode"
    description: str = "Use this to scan for a QR code or a barcode in the current webcam view."
    args_schema: Type[ScanCodeInput] = ScanCodeInput
    
    def _run(self) -> str:
        try:
            current_frame = get_current_frame()
            if current_frame is None: 
                return "Error: Could not get an image from the webcam."
            decoded_objects = decode(current_frame)
            if not decoded_objects: 
                return "I looked, but I couldn't find any QR codes or barcodes to scan."
            results = [f"Found a {obj.type} with data: '{obj.data.decode('utf-8')}'" for obj in decoded_objects]
            return " | ".join(results)
        except Exception as e:
            return f"Sorry, an error occurred while scanning for codes: {e}"

class AppLauncherInput(BaseModel):
    app_name: str = Field(description="The name of the application to open.")

class OpenApplicationTool(BaseTool):
    name: str = "open_application"
    description: str = "Use this to open applications like 'notepad', 'chrome', 'calculator', or 'explorer'."
    args_schema: Type[AppLauncherInput] = AppLauncherInput
    known_apps: ClassVar[dict] = {
        "notepad": "notepad.exe", 
        "chrome": "chrome.exe", 
        "calculator": "calc.exe", 
        "explorer": "explorer.exe"
    }
    
    def _run(self, app_name: str) -> str:
        app_name = app_name.lower()
        if app_name not in self.known_apps:
            return f"Sorry, I don't know how to open '{app_name}'. I can open: {', '.join(self.known_apps.keys())}."
        try:
            subprocess.Popen(["start", self.known_apps[app_name]], shell=True)
            return f"I've opened {app_name} for you."
        except Exception as e:
            return f"Sorry, an error occurred while opening the application: {e}"

class WeatherInput(BaseModel):
    city: str = Field(description="The city name to get weather for.")

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Use this to get current weather information for a city."
    args_schema: Type[WeatherInput] = WeatherInput
    
    def _run(self, city: str) -> str:
        extracted_city = extract_city_from_query(city)
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key: 
            return "Error: OpenWeatherMap API key is not set."
        
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": extracted_city, "appid": api_key, "units": "metric"}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            main = data['main']
            weather_desc = data['weather'][0]['description']
            return (f"The current weather in {extracted_city} is {main['temp']}°C with {weather_desc}. "
                    f"It feels like {main['feels_like']}°C, and the humidity is {main['humidity']}%.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404: 
                return f"Sorry, I couldn't find the city '{extracted_city}'."
            return f"Sorry, there was an HTTP error: {e}"
        except Exception as e: 
            return f"An error occurred fetching the weather: {e}"

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")

class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search for current events, news, sports scores, or general information."
    args_schema: Type[WebSearchInput] = WebSearchInput
    _base_search_tool = TavilySearch(max_results=3)
    
    def _run(self, query: str) -> str:
        try:
            cleaned_query = clean_query(query)
            
            if any(keyword in cleaned_query.lower() for keyword in ["cricket", "score", "match"]):
                enhanced_query = f"{cleaned_query} live score today"
            elif any(keyword in cleaned_query.lower() for keyword in ["news", "latest"]):
                enhanced_query = f"latest India news {cleaned_query}"
            else:
                enhanced_query = cleaned_query
            
            print(f"Searching for: '{enhanced_query}'")
            raw_results = self._base_search_tool.run(enhanced_query)
            
            formatted_results = format_search_results_safely(raw_results)
            return formatted_results
            
        except Exception as e:
            return f"Sorry, I encountered an error while searching: {str(e)}"

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

class RememberThisInput(BaseModel):
    fact_to_remember: str = Field(description="Information to remember.")

class RememberThisTool(BaseTool):
    name: str = "remember_this"
    description: str = "Remember specific information for later recall."
    args_schema: Type[RememberThisInput] = RememberThisInput
    
    def _run(self, fact_to_remember: str) -> str:
        return save_memory(fact_to_remember)

class RecallInfoInput(BaseModel):
    query: str = Field(description="Query to search memories.")

class RecallInfoTool(BaseTool):
    name: str = "recall_information"
    description: str = "Search previously remembered information."
    args_schema: Type[RecallInfoInput] = RecallInfoInput
    
    def _run(self, query: str) -> str:
        memories = recall_memories(query)
        if not memories: 
            return "I don't have any memories relevant to that question."
        return "Here are some relevant memories I found:\n- " + "\n- ".join(memories)

# --- Initialize Tool Instances ---
analyze_image_with_query = AnalyzeImageWithQueryTool()
play_youtube_song = PlayYouTubeSongTool()
read_text_from_image = ReadTextTool()
scan_qr_or_barcode = ScanCodeTool()
open_application = OpenApplicationTool()
get_weather = GetWeatherTool()
tavily_web_search = TavilySearchTool()
write_note = WriteNoteTool()
remember_this = RememberThisTool()
recall_information = RecallInfoTool()