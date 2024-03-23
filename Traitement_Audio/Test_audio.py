import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('environmental_sound_recognition_model.h5')

# Function to preprocess audio file for prediction
def preprocess_audio(audio_file_path):
    # Preprocess the audio file (you may need to adjust this based on your audio data)
    # Here, we're assuming the audio file is in a format compatible with your model input
    # You may need to convert the audio file to a spectrogram or other suitable representation
    # before passing it to the model for prediction.
    # For simplicity, we'll just resize the audio file to match the input size expected by the model.
    audio_data = image.load_img(audio_file_path, target_size=(64, 64))
    audio_array = image.img_to_array(audio_data)
    audio_array = np.expand_dims(audio_array, axis=0)
    return audio_array

# Example usage: Predict the class of a single audio file
audio_file_path = 'test.wav'
preprocessed_audio = preprocess_audio(audio_file_path)
predictions = model.predict(preprocessed_audio)
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)