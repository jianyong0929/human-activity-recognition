import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
def webcam():
    try:
        subprocess.run(['python', 'webcam.py'])
    except Exception as e:
        messagebox.showerror("Error", str(e))


def video():
    try:
        subprocess.run(['python', 'video_input.py'])
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create a label for the welcome text
window = tk.Tk()
window.title("Human Activity Recognition")
window.geometry('650x300')  # Adjusted the width
# Background color
window.configure(bg='#2C3E50')

# Create a label for the welcome text
welcome_label = ttk.Label(window, text="Human Activity Recognition", font=("Helvetica", 20, "bold"), foreground="white", background="#2C3E50")
welcome_label.grid(row=0, column=0, pady=20, columnspan=2)  # Used grid instead of pack

# Create a frame for the buttons
frame = ttk.Frame(window, padding=(20, 10))
frame.grid(row=1, column=0, pady=50, columnspan=2)

# Create "Start Webcam" button
button_1= tk.Button(frame, text="Real-Time Activity Recognition", command=webcam, bg='#3498DB', fg='white', padx=20, pady=10, bd=0, font=("Helvetica", 12, "bold"), relief=tk.FLAT)
button_1.grid(row=0, column=0)  # Used grid instead of pack

# Create "Stop Webcam" button
button_2 = tk.Button(frame, text="Activity Recognition From Video", command=video, bg='#E74C3C', fg='white', padx=20, pady=10, bd=0, font=("Helvetica", 12, "bold"), relief=tk.FLAT)
button_2.grid(row=0, column=1, padx=20)  # Used grid instead of pack

# Run the GUI
window.mainloop()