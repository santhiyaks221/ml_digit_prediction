from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

def launch_gui():
    window = Tk()
    window.title("Handwritten Digit Recognition")

    # Set a custom font and window background
    window.configure(bg="#2c3e50")

    # Create a frame for the canvas and labels
    frame = Frame(window, bg='#34495e', bd=5)
    frame.pack(pady=20)

    # Update canvas style
    canvas = Canvas(frame, width=350, height=290, bg='black', bd=2, relief=RIDGE)
    canvas.grid(row=0, column=0, padx=20, pady=20)

    # Add title label
    title = Label(window, text="Draw a Digit", font=('Helvetica', 20, 'bold'), bg="#2c3e50", fg="white")
    title.pack()

    # Add result label with styling
    l1 = Label(window, text="", font=('Helvetica', 20), bg="#2c3e50", fg="#e74c3c")
    l1.pack(pady=20)

    # Function to clear the canvas
    def clear_canvas():
        canvas.delete("all")
        l1.config(text="")

    # Function to predict the digit
    def predict_digit():
        x = window.winfo_rootx() + canvas.winfo_x()
        y = window.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28)).convert('L')
        vec = np.array(img).flatten() / 255.0  # Normalize

        # Load Theta values
        Theta1 = np.loadtxt('Theta1.txt')
        Theta2 = np.loadtxt('Theta2.txt')

        pred = predict(Theta1, Theta2, vec[np.newaxis, :])
        l1.config(text="Digit = " + str(pred[0]))

    # Styling buttons with custom background colors and fonts
    button_frame = Frame(window, bg="#2c3e50")
    button_frame.pack(pady=10)

    clear_btn = Button(button_frame, text="Clear Canvas", command=clear_canvas, bg="#e67e22", fg="white", font=('Helvetica', 12, 'bold'), padx=20, pady=10, relief=FLAT)
    clear_btn.grid(row=0, column=0, padx=10)

    predict_btn = Button(button_frame, text="Predict Digit", command=predict_digit, bg="#27ae60", fg="white", font=('Helvetica', 12, 'bold'), padx=20, pady=10, relief=FLAT)
    predict_btn.grid(row=0, column=1, padx=10)

    # Initialize lastx and lasty
    global lastx, lasty
    lastx, lasty = None, None

    # Function to draw on canvas
    def draw(event):
        global lastx, lasty
        x, y = event.x, event.y
        if lastx is not None and lasty is not None:
            canvas.create_line((lastx, lasty, x, y), width=15, fill='white', capstyle=ROUND)
        lastx, lasty = x, y

    # Reset drawing state
    def reset(event):
        global lastx, lasty
        lastx, lasty = None, None

    # Bind canvas events
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", reset)

    # Adjust window size and position
    window.geometry("550x600")
    window.mainloop()

if __name__ == "__main__":
    launch_gui()
