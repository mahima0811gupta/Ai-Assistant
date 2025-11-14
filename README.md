Dora – Your Personal AI Assistant

Dora is an advanced AI-powered personal assistant that integrates speech recognition, text-to-speech, computer vision, and multimodal intelligence using Google Gemini, Groq, and LangChain frameworks.
It listens to your voice, understands your commands, interacts with the webcam, and responds in real time — just like a human assistant.

🚀 Features
🗣️ Voice Interaction

Real-time speech-to-text using Gemini API.

Detects when you start and stop speaking automatically using Silero Voice Activity Detection (VAD).

Converts AI responses back into natural human speech using Google Text-to-Speech (gTTS).

📷 Computer Vision

Live webcam integration using OpenCV.

Can analyze objects, read text, detect QR/barcodes, and describe what’s visible through the camera using Gemini Vision.

💬 Conversational Intelligence

Powered by LangChain, LangGraph, and Groq for reasoning and natural language understanding.

Maintains a chat history using memory management tools for contextual responses.

Can recall saved information and write or append notes.

🌦️ Smart Utilities

Fetches live weather updates via the OpenWeather API.

Can search the web for real-time news or answers using Tavily Search.

Plays music or videos directly on YouTube via voice command.

Opens common desktop applications like Notepad, Calculator, and Chrome.

🔊 Real-Time Feedback

Gradio-based graphical interface for live webcam display, chat history, and voice playback.

Automatically records and stops recording based on silence detection.

🧠 Tech Stack
Category	Technologies Used
Programming Language	Python
Core Libraries	Gradio, OpenCV, Torch, NumPy, Pydub, SoundDevice
AI Models	Gemini 1.5 Flash (Text + Vision), Silero VAD
Frameworks	LangChain, LangGraph
Speech	Google gTTS (Text-to-Speech), Gemini (Speech-to-Text)
Web Search	Tavily, Requests
Other Tools	PyWhatKit, Pytesseract, PyZbar (QR Detection)
🏗️ Project Architecture
E:/CUBA/
│
├── main.py                   # Main Gradio app with webcam and voice logic
├── ai_agent.py               # AI reasoning and response generation
├── speech_to_text.py         # Gemini-based audio transcription
├── text_to_speech.py         # Converts AI replies into voice using gTTS
├── tools.py                  # Vision, weather, search, and utility tools
├── shared_frame.py           # Shares webcam frames across modules
├── memory_manager.py         # Stores and recalls user memories
├── requirements.txt          # All dependencies
├── .gitignore                # Ensures sensitive and unnecessary files are ignored
└── .env                      # (Local only) Stores your API keys securely

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/mahima0811gupta/Ai-Assistant.git
cd Ai-Assistant

2️⃣ Create a Virtual Environment
python -m venv webcam_env
webcam_env\Scripts\activate

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Create a .env File

Inside the project folder, create a file named .env and add:

GOOGLE_API_KEY=your_google_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
GROQ_API_KEY=your_groq_api_key_here

5️⃣ Run the Application
python main.py


Once it starts, open the local Gradio link in your browser to interact with Dora.

💡 Example Commands

You can ask Dora things like:

“What’s the weather in Delhi?”

“Describe what you see in the webcam.”

“Play Tum Hi Ho on YouTube.”

“Open Notepad.”

“Remember my name is Mahima.”

“Read the text in front of me.”

“What’s in this image?”

“Tell me today’s news.”

🔒 Security Note

The .env file must never be uploaded to GitHub (contains your private API keys).

The .gitignore already includes .env, /webcam_env/, and cache folders to prevent accidental leaks.

📄 Future Enhancements

🌍 Multilingual speech support (Hindi, French, etc.)

🤖 Integration with emotion recognition and gesture tracking

🧠 Local model inference for offline use

📱 Mobile-friendly or web dashboard deployment
