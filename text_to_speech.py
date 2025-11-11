
import os
from gtts import gTTS
import pygame
import time

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Converts a string of text to an MP3 file using gTTS and then plays it using Pygame.
    
    Args:
        input_text (str): The text to be converted to speech.
        output_filepath (str): The path to save the generated MP3 file.
    """
    try:
        # 1. Generate the audio file from text using Google Text-to-Speech
        language = "en"
        audio_obj = gTTS(text=input_text, lang=language, slow=False)
        audio_obj.save(output_filepath)
        print(f"Audio saved to {output_filepath}")

        # 2. Play the generated audio file using Pygame
        pygame.mixer.init()
        pygame.mixer.music.load(output_filepath)
        pygame.mixer.music.play()
        
        # 3. Wait for the audio to finish playing
        # This loop uses Pygame's clock to prevent the script from exiting prematurely.
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10) # Check 10 times per second
        
        # 4. Clean up the mixer
        pygame.mixer.quit()
        print("Playback finished.")
        
        # 5. Optional: Clean up the created audio file
        # Uncomment the line below if you want to delete the file after playing
        # os.remove(output_filepath)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: This part will only run when you execute `python text_to_speech.py`
if __name__ == "__main__":
    # Initialize pygame to get the welcome message out of the way
    pygame.init()

    # Example text and file
    example_text = "Hello, this is a test of the text-to-speech system."
    example_filepath = "test_audio.mp3"
    
    # Run the function
    text_to_speech_with_gtts(example_text, example_filepath)