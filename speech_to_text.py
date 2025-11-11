

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini client from your existing setup
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini in speech_to_text.py: {e}")

def transcribe_with_gemini(audio_filepath):
    """
    Transcribes audio using the Gemini API by uploading the file first.
    """
    if not os.path.exists(audio_filepath):
        print(f"Error: Audio file not found at {audio_filepath}")
        return ""
        
    print(f"Uploading audio file: {audio_filepath} to Gemini...")
    try:
        # 1. Upload the audio file to the Gemini API
        audio_file = genai.upload_file(path=audio_filepath)
        print("File uploaded successfully.")

        # 2. Ask the Gemini model to transcribe the audio
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Transcribe the following audio file accurately."
        response = model.generate_content([prompt, audio_file])

        # 3. Clean up the uploaded file to avoid cluttering your account
        genai.delete_file(audio_file.name)
        print("Uploaded file cleaned up.")

        transcribed_text = response.text.strip()
        print(f"Transcription successful: {transcribed_text}")
        return transcribed_text

    except Exception as e:
        print(f"An error occurred during Gemini transcription: {e}")
        return ""
    

# audio_filepath = "test_text_to_speech.mp3"
# print(transcribe_with_gemini(audio_filepath))
