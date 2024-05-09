from tkinter import filedialog
import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import tkinter as tk
from threading import Thread
from tkinter import ttk
import json

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load your trained model
model = load_model('activity_recognition_model'
                   '.h5')

def browse_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def upload_file():
    file_path = file_entry.get()
    if not file_path:
        error_label.config(text="Error: Please select a file.")
    else:
        error_label.config(text="")  # Clear any previous error message
        process_input_video(model, file_path)

# def get_activity_by_label(label):
#     keypoints_base_path = r'..\Keypoints'
#     activity_labels = {activity: idx for idx, activity in enumerate(sorted(os.listdir(keypoints_base_path)))}
#     for activity, activity_label in activity_labels.items():
#         if activity_label == label:
#             return activity


def get_activity_by_label(label):
    # We open the 'activities.json' file in read mode ('r')
    with open('activities.json', 'r') as f:
        # We load our dictionary from the JSON file
        activity_labels = json.load(f)

    # We look for the 'toy' with the given 'label'
    for activity, activity_label in activity_labels.items():
        if activity_label == label:
            # If we find it, we return the 'toy'
            return activity


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    return image

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)  # 33 landmarks, each with x, y, z, and visibility
    return pose

def standardize_frames(keypoints, desired_length=30):
    frame_count = keypoints.shape[0]
    if frame_count > desired_length:
        indices = np.linspace(0, frame_count-1, desired_length, dtype=int)
        standardized_keypoints = keypoints[indices]
    elif frame_count < desired_length:
        padding = np.zeros((desired_length - frame_count, keypoints.shape[1]))
        standardized_keypoints = np.vstack((keypoints, padding))
    else:
        standardized_keypoints = keypoints
    return standardized_keypoints

def process_input_video(model, video_file_path):
    global webcam_running
    frame_num = 0
    activity = "Detecting....."
    threshold = 0.9
    sequence = []
    cap = cv2.VideoCapture(video_file_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Draw landmarks on the image
            image = draw_landmarks(image, results)

            # Every 30 frames, make a prediction
            if len(sequence) == 30 and frame_num % 30 == 0:
                standardized_sequence = standardize_frames(np.array(sequence))
                prediction = model.predict(standardized_sequence[np.newaxis, ...])
                predicted_activity = np.argmax(prediction)
                # Check if the maximum prediction probability is above the threshold
                if prediction[0][predicted_activity] > threshold:
                    activity = get_activity_by_label(predicted_activity)
                else:
                    activity = "Detecting....."
                print(f"Activity prediction: {activity}")

            # Display the recognized activity on the image
            cv2.putText(image, f"Activity: {activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Video Input', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

# Create the main window
upload_window = tk.Tk()
upload_window.title("Activity Recognition from video")
upload_window.geometry('500x300')
upload_window.configure(bg='#2C3E50')  # Background color

# Create a label for the welcome text
welcome_label = ttk.Label(upload_window, text="Activity Recognition from video", font=("Helvetica", 20, "bold"), foreground="white", background="#2C3E50")
welcome_label.pack(pady=20)

# Create a frame for file selection
file_frame = ttk.Frame(upload_window, padding=(20, 10))
file_frame.pack(pady=20)

# Create an entry widget to display the selected file path
file_entry = ttk.Entry(file_frame, font=("Helvetica", 12))
file_entry.grid(row=0, column=0, padx=10, pady=10)

# Create "Browse" button for file selection
browse_button = tk.Button(file_frame, text="Browse", command=browse_file, bg='#3498DB', fg='white', padx=10, pady=5, bd=0, font=("Helvetica", 10, "bold"), relief=tk.FLAT)
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Create a frame for the upload button
upload_frame = ttk.Frame(upload_window, padding=(20, 10))
upload_frame.pack(pady=20)

# Create "Upload" button
upload_button = tk.Button(upload_frame, text="Upload", command=upload_file, bg='#2ECC71', fg='white', padx=20, pady=10, bd=0, font=("Helvetica", 12, "bold"), relief=tk.FLAT)
upload_button.pack()

# Create a label for error messages
error_label = ttk.Label(upload_window, text="", foreground="red", background="#2C3E50")
error_label.pack(pady=10)

# Run the GUI
upload_window.mainloop()
