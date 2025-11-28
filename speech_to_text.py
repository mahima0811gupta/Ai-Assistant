

# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# # Configure the Gemini client from your existing setup
# try:
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# except Exception as e:
#     print(f"Error configuring Gemini in speech_to_text.py: {e}")

# def transcribe_with_gemini(audio_filepath):
#     """
#     Transcribes audio using the Gemini API by uploading the file first.
#     """
#     if not os.path.exists(audio_filepath):
#         print(f"Error: Audio file not found at {audio_filepath}")
#         return ""
        
#     print(f"Uploading audio file: {audio_filepath} to Gemini...")
#     try:
#         # 1. Upload the audio file to the Gemini API
#         audio_file = genai.upload_file(path=audio_filepath)
#         print("File uploaded successfully.")

#         # 2. Ask the Gemini model to transcribe the audio
#         model = genai.GenerativeModel('gemini-2.5-flash')
        
#         prompt = "Transcribe the following audio file accurately."
#         response = model.generate_content([prompt, audio_file])

#         # 3. Clean up the uploaded file to avoid cluttering your account
#         genai.delete_file(audio_file.name)
#         print("Uploaded file cleaned up.")

#         transcribed_text = response.text.strip()
#         print(f"Transcription successful: {transcribed_text}")
#         return transcribed_text

#     except Exception as e:
#         print(f"An error occurred during Gemini transcription: {e}")
#         return ""
    

# # audio_filepath = "test_text_to_speech.mp3"
# # print(transcribe_with_gemini(audio_filepath))



import os
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

def transcribe_with_deepgram(audio_filepath):
    """
    Transcribes audio using Deepgram API for fast, accurate transcription.
    
    Args:
        audio_filepath (str): Path to the audio file (WAV, MP3, etc.)
    
    Returns:
        str: Transcribed text or empty string on error
    """
    if not os.path.exists(audio_filepath):
        print(f"❌ Error: Audio file not found at {audio_filepath}")
        return ""
    
    # Get API key from environment
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("❌ Error: DEEPGRAM_API_KEY not found in .env file")
        return ""
    
    try:
        print(f"🎙️ Transcribing with Deepgram: {audio_filepath}")
        
        # Initialize Deepgram client with API key
        deepgram = DeepgramClient(api_key=api_key)
        
        # Read audio file
        with open(audio_filepath, "rb") as audio_file:
            buffer_data = audio_file.read()
        
        # Create payload
        payload = {
            "buffer": buffer_data
        }
        
        # Configure transcription options
        options = {
            "model": "nova-2",
            "smart_format": True,
            "language": "en",
            "punctuate": True,
        }
        
        # Transcribe the audio
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, 
            options
        )
        
        # Extract transcription text
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        if transcript:
            print(f"✅ Deepgram transcription: {transcript}")
            return transcript.strip()
        else:
            print("⚠️ Warning: Empty transcription received")
            return ""
            
    except AttributeError as e:
        print(f"❌ Deepgram SDK version mismatch: {e}")
        print("💡 Try: pip install deepgram-sdk==3.5.0 --upgrade")
        return ""
    except Exception as e:
        print(f"❌ Deepgram transcription error: {e}")
        return ""


# Fallback function using Gemini (optional)
def transcribe_with_gemini_fallback(audio_filepath):
    """
    Fallback transcription using Gemini API.
    Use this if Deepgram fails or for specific use cases.
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        if not os.path.exists(audio_filepath):
            print(f"❌ Error: Audio file not found at {audio_filepath}")
            return ""
            
        print(f"🎙️ Transcribing with Gemini (fallback): {audio_filepath}")
        
        audio_file = genai.upload_file(path=audio_filepath)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Transcribe the following audio file accurately."
        response = model.generate_content([prompt, audio_file])
        
        genai.delete_file(audio_file.name)
        
        transcribed_text = response.text.strip()
        print(f"✅ Gemini transcription: {transcribed_text}")
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Gemini transcription error: {e}")
        return ""


# Main function - keeps original name for backward compatibility
def transcribe_with_gemini(audio_filepath):
    """
    Main transcription function using Deepgram (faster) with Gemini fallback.
    
    Args:
        audio_filepath (str): Path to audio file
    
    Returns:
        str: Transcribed text
    """
    # Try Deepgram first (much faster)
    result = transcribe_with_deepgram(audio_filepath)
    
    # If Deepgram fails, fallback to Gemini
    if not result:
        print("⚠️ Deepgram failed, using Gemini fallback...")
        result = transcribe_with_gemini_fallback(audio_filepath)
    
    return result