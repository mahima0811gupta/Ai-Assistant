Dora – Your Personal AI Assistant

Dora is an advanced voice-controlled AI assistant powered by Google Gemini, LangChain, and LangGraph. It listens to your voice, understands your commands, performs real-time actions, speaks back to you, remembers information, opens applications, plays music, fetches weather, searches the web, and more.

Dora works like a smart conversational AI — fully voice-driven, without requiring any webcam or visual input.

🚀 Features
🗣️ Voice Interaction

Real-time speech-to-text using Gemini API

Automatic Voice Activity Detection (Silero VAD)

Converts AI responses into natural speech using gTTS

Smooth, hands-free interaction

💬 Conversational Intelligence

Powered by LangChain, LangGraph, and Gemini Flash

Maintains chat context and long-term memory

Uses memory tools to remember facts you say

Recalls previous information when asked

🧠 Smart Utilities

Fetches live weather updates

Searches the web using Tavily Search

Plays YouTube music instantly

Opens desktop applications:

Notepad

Chrome

Calculator

File Explorer

Creates and appends notes via voice

Understands everyday commands naturally

🔊 Real-Time Feedback

Clean Gradio UI with:

Chat interface

Voice input/output

Status indicators

🧠 Tech Stack
Category	Technologies
Programming Language	Python
Core Libraries	Gradio, Torch, NumPy, Pydub, SoundDevice
AI Models	Gemini 2.5 Flash (Text), Silero VAD
Frameworks	LangChain, LangGraph
Speech	Gemini (STT), gTTS (TTS)
Web Search	Tavily, Requests
Utilities	PyWhatKit
🏗️ Project Structure
E:/CUBA/
│── main.py                # Main Gradio app (voice)
│── ai_agent.py            # AI reasoning + tool routing
│── speech_to_text.py      # Gemini audio transcription
│── text_to_speech.py      # AI → Speech using gTTS
│── tools.py               # YouTube, search, apps, notes, memory
│── memory_manager.py       # Saves & recalls user memories
│── requirements.txt       # Project dependencies
│── .gitignore             # Excludes sensitive files
└── .env                   # API keys (NOT uploaded)

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/mahima0811gupta/Ai-Assistant.git
cd Ai-Assistant

2️⃣ Create Virtual Environment
python -m venv webcam_env
webcam_env\Scripts\activate

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Create .env File

Add your keys:

GOOGLE_API_KEY=your_google_key
OPENWEATHER_API_KEY=your_weather_key
TAVILY_API_KEY=your_tavily_key


5️⃣ Run the Application
python main.py

💡 Example Voice Commands

You can say:

“What’s the weather in Delhi?”

“Play Tum Hi Ho on YouTube.”

“Open Notepad.”

“Remember that my name is Mahima.”

“Tell me today’s news.”

“Write a note saying I have a meeting tomorrow.”

“Search for India cricket score.”

🔒 Security Note

.env is ignored and NOT uploaded

Never share API keys publicly

The application runs locally, keeping your data private

📄 Future Enhancements

Multilingual voice support

Offline speech recognition

Better long-term memory

Mobile app version
