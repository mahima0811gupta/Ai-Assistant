# shared_frame.py
# This module holds the shared state of the webcam frame to avoid circular imports.

# This global variable will hold the most recent frame from the webcam.
current_webcam_frame = None

def set_current_frame(image_frame):
    """Sets the global variable to the current webcam frame."""
    global current_webcam_frame
    current_webcam_frame = image_frame

def get_current_frame():
    """Gets the current webcam frame."""
    global current_webcam_frame
    return current_webcam_frame




