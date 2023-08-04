import PySimpleGUI as sg
import subprocess
import io
from PIL import Image
import os
import cv2 as cv
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from results import diagnosis

def get_results(selected_file):
    img = cv.imread(selected_file)
    res = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_CUBIC)
    test = np.array([res])
    model = keras.models.load_model("./savedmodel/")
    prediction = np.argmax(model.predict(test))
    result = diagnosis(prediction)
    return result

def main():
    sg.theme('LightGreen10')


    layout = [
        [sg.Text('Select a file to view:')],
        [sg.Input(key='-FILE-', enable_events=True, size=(40, 40)), sg.FileBrowse()],
        [sg.Button('Next')]
    ]

    window = sg.Window('Ingredient Defender - Select File', layout, resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Next':
            selected_file = values['-FILE-']
            if selected_file:
                window.close()  # Close the current window

                script_dir = os.path.dirname(os.path.realpath(__file__))
                relative_file = os.path.relpath(selected_file, script_dir)

                # Get the output from results.py based on the selected file
                result = get_results(relative_file)

                # Create a new window to display results and the image
                layout_result = [
                    [sg.Text('Diagnosis:')],
                    [sg.Multiline(result, size=(120, 10), disabled=True)],
                    [sg.Image(key='-IMAGE-', size=(200, 200))],
                    [sg.Button('Exit')]
                ]
                window_result = sg.Window('Ingredient Defender - Results', layout_result, resizable=True, finalize=True)

                # Display the selected file's image (modify this part based on your image display logic)
                with open(selected_file, 'rb') as f:
                    image_bytes = f.read()
                image = Image.open(io.BytesIO(image_bytes))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window_result['-IMAGE-'].update(data=bio.getvalue())

                while True:
                    event_result, _ = window_result.read()

                    if event_result in (sg.WIN_CLOSED, 'Exit'):
                        break

                window_result.close()

    window.close()

main()