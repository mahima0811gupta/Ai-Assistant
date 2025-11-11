




import os
import gradio as gr
import cv2
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import time
import asyncio
import logging

# --- Local Imports ---
from speech_to_text import transcribe_with_gemini
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_gtts
from shared_frame import set_current_frame

# Configure clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# SECTION 1: VOICE ACTIVITY DETECTION (VAD) RECORDER 

class VoiceRecorder:
    """
    Handles real-time audio recording from the microphone using Voice Activity Detection (VAD).
    """
    def __init__(self):
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            print("✅ Silero VAD model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: VAD model loading error: {e}. Voice recording will use a fallback.")
            self.model = None
        self.is_recording = False

    def record_until_silence(self, sample_rate=16000, silence_duration_s=1.5, min_speech_duration_s=0.25):
        """Records audio and automatically stops after a specified duration of silence."""
        if not self.model:
            return self._fallback_record(sample_rate)
            
        self.is_recording = True
        recorded_audio = []
        silent_chunks = 0
        speech_started = False
        
        chunk_size = 512
        max_silent_chunks = int(silence_duration_s * (sample_rate / chunk_size))
        
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=chunk_size)
            with stream:
                print("🎙️ Listening...")
                while self.is_recording:
                    audio_chunk, _ = stream.read(chunk_size)
                    audio_chunk_np = audio_chunk.flatten()
                    speech_prob = self.model(torch.from_numpy(audio_chunk_np), sample_rate).item()
                    
                    if speech_prob > 0.5:
                        speech_started = True
                        silent_chunks = 0
                        recorded_audio.append(audio_chunk_np)
                    elif speech_started:
                        silent_chunks += 1
                        recorded_audio.append(audio_chunk_np)
                        if silent_chunks > max_silent_chunks:
                            break
            print("🛑 Recording stopped.")
        except Exception as e:
            print(f"❌ Fatal audio stream error: {e}")
            return None
            
        if not recorded_audio:
            print("🎤 No speech detected.")
            return None

        full_audio = np.concatenate(recorded_audio)
        if len(full_audio) < sample_rate * min_speech_duration_s:
            print("🎤 Recording was too short, ignoring.")
            return None

        output_filepath = "user_speech.wav"
        try:
            wav.write(output_filepath, sample_rate, (full_audio * 32767).astype(np.int16))
            return output_filepath
        except Exception as e:
            print(f"❌ Error saving recorded audio file: {e}")
            return None
    
    def _fallback_record(self, sample_rate=16000, duration=5):
        """Fallback recording for a fixed duration if VAD is not available."""
        try:
            print(f"🎙️ VAD not found. Recording for a fixed {duration} seconds...")
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            output_filepath = "user_speech.wav"
            wav.write(output_filepath, sample_rate, (recording.flatten() * 32767).astype(np.int16))
            print("🛑 Fallback recording finished.")
            return output_filepath
        except Exception as e:
            print(f"❌ Error during fallback recording: {e}")
            return None


# Lazy Initialization for the Recorder 
_recorder_instance = None

def get_recorder():
    """Initializes the VoiceRecorder only when it's first needed."""
    global _recorder_instance
    if _recorder_instance is None:
        print("Initializing VoiceRecorder for the first time...")
        _recorder_instance = VoiceRecorder()
    return _recorder_instance


# SECTION 2: WEBCAM & MAIN APPLICATION LOGIC 

camera = None

def start_webcam():
    """Initializes and holds the webcam resource, trying multiple indices."""
    global camera
    if camera is None:
        for idx in range(4):
            try:
                camera_candidate = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if camera_candidate and camera_candidate.isOpened():
                    ret, _ = camera_candidate.read()
                    if ret:
                        camera = camera_candidate
                        print(f"✅ Camera {idx} initialized successfully")
                        break
                    else:
                        camera_candidate.release()
            except Exception as e:
                print(f"Camera {idx} failed to open: {e}")
        if camera is None:
            print("❌ Error: Could not start any camera.")
    return get_webcam_frame() 

def get_webcam_frame():
    """Captures a frame from the running webcam and shares it."""
    global camera
    if camera and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            set_current_frame(rgb_frame)
            return rgb_frame
    return None

def listen_and_respond(chat_history):
    """Primary function triggered by the 'Click to Speak' button."""
    yield chat_history, None, "Listening...", gr.update(interactive=False)
    try:
        recorder = get_recorder() 
        user_audio_path = recorder.record_until_silence()
    except Exception as e:
        print(f"❌ An unexpected error occurred during recording: {e}")
        user_audio_path = None

    if user_audio_path is None:
        yield chat_history, None, "I didn't hear anything. Click to try again.", gr.update(interactive=True)
        return

    for state in process_recorded_audio(chat_history, user_audio_path):
        yield state

def process_recorded_audio(chat_history, user_audio_path):
    """Handles the full pipeline after audio is recorded with error handling."""
    chat_history.append({"role": "user", "content": "(Audio captured...)"})
    yield chat_history, None, "Transcribing...", gr.update(interactive=False)
    user_text = transcribe_with_gemini(user_audio_path)
        
    if not user_text:
        chat_history.pop()
        yield chat_history, None, "Could not understand your speech. Click to try again.", gr.update(interactive=True)
        return

    chat_history[-1] = {"role": "user", "content": user_text}
    yield chat_history, None, "Dora is thinking...", gr.update(interactive=False)

    response_text = ask_agent(user_query=user_text, chat_history=chat_history)

    yield chat_history, None, "Generating speech...", gr.update(interactive=False)
    
    mp3_path = "response.mp3"
    wav_path = "response.wav"
    response_audio_file = None
    
    try:
        text_to_speech_with_gtts(input_text=response_text, output_filepath=mp3_path)
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            response_audio_file = wav_path
        except Exception as e:
            print(f"⚠️ Error converting MP3 to WAV: {e}. Falling back to MP3.")
            response_audio_file = mp3_path
    except Exception as e:
        print(f"❌ Fatal TTS error: {e}")

    chat_history.append({"role": "assistant", "content": response_text})
    
    yield chat_history, response_audio_file, "Ready. Click the button to speak.", gr.update(interactive=True)


# SECTION 3: GRADIO USER INTERFACE 

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='color: orange; text-align: center; font-size: 4em;'>👧🏼 Dora – Your Personal AI Assistant</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            webcam_output = gr.Image(label="Live Feed", show_label=False, width=640, height=480)

        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            chatbot = gr.Chatbot(label="Conversation", height=400, show_label=False, type="messages")
            audio_out = gr.Audio(label="Assistant's Voice", autoplay=True, visible=False)
            speak_btn = gr.Button("🎤 Click to Speak", variant="primary")
            status_box = gr.Textbox(label="Status", value="Initializing...", interactive=False)
            clear_btn = gr.Button("Clear Chat", variant="secondary")

    gr.Timer(0.033).tick(fn=get_webcam_frame, outputs=[webcam_output])

    speak_btn.click(
        fn=listen_and_respond,
        inputs=[chatbot],
        outputs=[chatbot, audio_out, status_box, speak_btn]
    )

    clear_btn.click(
        fn=lambda: ([], None, "Ready. Click the button to speak."),
        outputs=[chatbot, audio_out, status_box]
    )
    
    demo.load(
        fn=start_webcam,
        outputs=[webcam_output]
    ).then(
        fn=lambda: "Ready. Click the button to speak.",
        outputs=[status_box]
    )


#SECTION 4: LAUNCH THE APPLICATION 

def silence_connection_reset(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        logging.debug("Ignored ConnectionResetError: Remote host closed connection.")
        return
    loop.default_exception_handler(context)

# Set event loop policy for better stability on Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
loop = asyncio.get_event_loop()
loop.set_exception_handler(silence_connection_reset)

if __name__ == "__main__":
    demo.launch(
        share=False,
        debug=False
    )
