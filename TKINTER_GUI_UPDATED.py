import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageEnhance
import threading
import os

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Attempt to import YOLOv5; ensure yolov5 is in your PYTHONPATH or installed correctly
try:
    import yolov5
    from yolov5 import YOLOv5
except ImportError:
    # Alternative: Load via torch hub if yolov5 is not a package
    from torch.hub import load as torch_load
    YOLOv5 = None  # Placeholder; will use torch hub

# Initialize Tkinter root
root = tk.Tk()
root.geometry("1200x700")
root.title("Ishan Malik's Image Drawing and Video Analytics Tool")
root.config(bg="white")

# Global Variables
pen_color = "black"
pen_size = 5
file_path = ""
video_capture = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

shape_type = "line"
a, b = None, None
x, y = None, None

# Keep a reference to images to prevent garbage collection
image_list = []

# Load Tumor Detection Model
def load_tumor_model():
    global tumor_model
    model_path = "C:/Users/Ishan/Downloads/Brain_Tumor_Model.h5"  # Update with your model path
    if not os.path.exists(model_path):
        messagebox.showerror("Model Not Found", f"Tumor detection model not found at {model_path}. Please provide the correct path.")
        tumor_model = None
        return
    try:
        tumor_model = load_model(model_path)
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Error loading tumor detection model: {e}")
        tumor_model = None

# Load YOLOv5 Model
def load_yolo_model():
    global yolo_model
    try:
        # Attempt to load via yolov5 package
        if YOLOv5:
            yolo_model = YOLOv5("yolov5s.pt", device='cpu')  # Change device if GPU is available
        else:
            # Load via torch hub
            yolo_model = torch_load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        messagebox.showerror("YOLOv5 Load Error", f"Error loading YOLOv5 model: {e}")
        yolo_model = None

# Initialize Models
load_tumor_model()
load_yolo_model()

# Initialize Tracker Variables
tracker = None
tracking = False

# Define Functions
def add_image():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="C:/Users/Ishan/OneDrive/Desktop/Important Documents/My Projects",
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
    )
    if file_path:
        try:
            image = Image.open(file_path).convert('RGB')
            width, height = int(image.width / 2), int(image.height / 2)
            image = image.resize((width, height), Image.LANCZOS)
            canvas.config(width=image.width, height=image.height)
            image_tk = ImageTk.PhotoImage(image)
            canvas.image = image_tk
            canvas.create_image(0, 0, image=image_tk, anchor="nw")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Error loading image: {e}")

def change_color():
    global pen_color
    color = colorchooser.askcolor(title="Select Pen Color")
    if color[1]:
        pen_color = color[1]

def change_size(size):
    global pen_size
    pen_size = size

def start_draw(event):
    global a, b
    a, b = event.x, event.y

def end_draw(event):
    global a, b
    a, b = None, None

def change_shape(shape):
    global shape_type
    shape_type = shape

def draw(event):
    global x, y, a, b
    x, y = event.x, event.y

    if shape_type == "line":
        draw_line(x, y, pen_size, pen_color)
    elif shape_type == "oval":
        draw_oval()
    elif shape_type == "rectangle":
        draw_rectangle()
    elif shape_type == "text":
        draw_text()

def draw_line(x, y, pen_size, pen_color):
    if a is not None and b is not None:
        canvas.create_line(a, b, x, y, fill=pen_color, width=pen_size)
    else:
        # Fallback to drawing a small oval if no previous point
        x1, y1 = (x - pen_size), (y - pen_size)
        x2, y2 = (x + pen_size), (y + pen_size)
        canvas.create_oval(x1, y1, x2, y2, fill=pen_color, outline='')

def draw_oval():
    if a is not None and b is not None and x is not None and y is not None:
        canvas.create_oval(a, b, x, y, outline=pen_color, width=pen_size)

def draw_rectangle():
    if a is not None and b is not None and x is not None and y is not None:
        canvas.create_rectangle(a, b, x, y, outline=pen_color, width=pen_size)

def draw_text():
    if a is not None and b is not None:
        text = "Jai Shri Krishna!"
        canvas.create_text(a, b, text=text, fill=pen_color, font=("Arial", 20))

def adjust_brightness(user_input1):
    if not file_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return
    try:
        factor = float(user_input1)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number for brightness.")
        return
    try:
        image = Image.open(file_path).convert('RGB')
        width, height = int(image.width / 2), int(image.height / 2)
        image = image.resize((width, height), Image.LANCZOS)
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(factor)
        canvas.config(width=image.width, height=image.height)
        image_tk = ImageTk.PhotoImage(image)
        canvas.image = image_tk
        canvas.create_image(0, 0, image=image_tk, anchor="nw")
    except Exception as e:
        messagebox.showerror("Brightness Adjustment Error", f"Error adjusting brightness: {e}")

def adjust_contrast(user_input2):
    if not file_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return
    try:
        factor = float(user_input2)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number for contrast.")
        return
    try:
        image = Image.open(file_path).convert('RGB')
        width, height = int(image.width / 2), int(image.height / 2)
        image = image.resize((width, height), Image.LANCZOS)
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(factor)
        canvas.config(width=image.width, height=image.height)
        image_tk = ImageTk.PhotoImage(image)
        canvas.image = image_tk
        canvas.create_image(0, 0, image=image_tk, anchor="nw")
    except Exception as e:
        messagebox.showerror("Contrast Adjustment Error", f"Error adjusting contrast: {e}")

def clear_canvas():
    canvas.delete("all")
    if file_path:
        try:
            image = Image.open(file_path).convert('RGB')
            width, height = int(image.width / 2), int(image.height / 2)
            image = image.resize((width, height), Image.LANCZOS)
            canvas.config(width=image.width, height=image.height)
            image_tk = ImageTk.PhotoImage(image)
            canvas.image = image_tk
            canvas.create_image(0, 0, image=image_tk, anchor="nw")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Error loading image: {e}")

def apply_filter(filter_type):
    if not file_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return
    try:
        image = Image.open(file_path).convert('RGB')
        width, height = int(image.width / 2), int(image.height / 2)
        image = image.resize((width, height), Image.LANCZOS)

        if filter_type == "Grayscale":
            image = ImageOps.grayscale(image)
        elif filter_type == "RGB":
            image = image.convert('RGB')  # Ensures image is in RGB mode
        elif filter_type == "Binary":
            image = image.convert('1')  # Convert to binary
        elif filter_type == "Blur":
            image = image.filter(ImageFilter.BLUR)
        elif filter_type == "Sharpen":
            image = image.filter(ImageFilter.SHARPEN)
        elif filter_type == "Smooth":
            image = image.filter(ImageFilter.SMOOTH)
        elif filter_type == "Emboss":
            image = image.filter(ImageFilter.EMBOSS)
        else:
            messagebox.showwarning("Unknown Filter", f"The filter '{filter_type}' is not recognized.")
            return

        canvas.config(width=image.width, height=image.height)
        image_tk = ImageTk.PhotoImage(image)
        canvas.image = image_tk
        canvas.create_image(0, 0, image=image_tk, anchor="nw")
    except Exception as e:
        messagebox.showerror("Filter Error", f"Error applying filter: {e}")

def detect_tumor():
    if not file_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return
    if not tumor_model:
        messagebox.showerror("Model Not Loaded", "Tumor detection model is not loaded.")
        return

    def run_detection():
        try:
            image = Image.open(file_path).convert('RGB')
            original_size = image.size

            # Preprocess the image
            img_array = keras_image.img_to_array(image)
            img_array = cv2.resize(img_array, (224, 224))  # Adjust based on model's expected input size
            img_array = img_array / 255.0  # Normalize if required by the model
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict using the model
            predictions = tumor_model.predict(img_array)

            # Process the output
            # Assuming the model outputs a segmentation mask
            mask = predictions[0]  # Get the first (and only) prediction
            mask = cv2.resize(mask, original_size)  # Resize mask to original image size
            mask = (mask > 0.5).astype('uint8') * 255  # Thresholding

            # Convert mask to image
            mask_image = Image.fromarray(mask).convert('L')  # Convert to grayscale image
            mask_image = mask_image.convert('RGBA')

            # Create a red overlay for the mask
            red_overlay = Image.new('RGBA', original_size, (255, 0, 0, 100))
            image_with_mask = Image.open(file_path).convert('RGBA')
            image_with_mask = Image.alpha_composite(image_with_mask, red_overlay)
            image_with_mask.putalpha(mask_image.split()[-1])  # Use mask as alpha channel

            # Display the image with mask
            width, height = int(image_with_mask.width / 2), int(image_with_mask.height / 2)
            image_with_mask = image_with_mask.resize((width, height), Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_with_mask)
            canvas.image = image_tk
            canvas.create_image(0, 0, image=image_tk, anchor="nw")
        except Exception as e:
            messagebox.showerror("Tumor Detection Error", f"Error during tumor detection: {e}")

    # Run detection in a separate thread to prevent GUI freezing
    threading.Thread(target=run_detection).start()

def load_video():
    global video_capture, tracking, tracker
    file_path = filedialog.askopenfilename(
        initialdir="C:/Users/Ishan/OneDrive/Desktop/Important Documents/My Projects",
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if file_path:
        video_capture = cv2.VideoCapture(file_path)
        if not video_capture.isOpened():
            messagebox.showerror("Video Load Error", "Error opening video file.")
            return
        tracking = False  # Reset tracking
        update_frame()

def start_tracking():
    global tracker, tracking
    if not video_capture:
        messagebox.showwarning("No Video", "Please load a video first.")
        return
    if tracking:
        messagebox.showinfo("Tracking", "Tracking is already in progress.")
        return
    tracking = True
    threading.Thread(target=initialize_tracker).start()

def initialize_tracker():
    global tracker, tracking
    try:
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Video Error", "Cannot read the first frame for tracking.")
            tracking = False
            return
        # Let the user select the ROI
        cv2.imshow("Select ROI", frame)
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        if bbox == (0,0,0,0):
            messagebox.showwarning("ROI Selection", "No ROI selected. Tracking canceled.")
            tracking = False
            return
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        update_frame()
    except Exception as e:
        messagebox.showerror("Tracking Initialization Error", f"Error initializing tracker: {e}")
        tracking = False

def update_frame():
    global video_capture, yolo_model, tracking, tracker
    if video_capture is not None:
        ret, frame = video_capture.read()
        if ret:
            # Perform object detection using YOLOv5
            if yolo_model:
                try:
                    if isinstance(yolo_model, YOLOv5):
                        results = yolo_model.predict(frame)
                        annotated_frame = results.render()[0]
                    else:
                        # If loaded via torch hub
                        results = yolo_model(frame)
                        annotated_frame = results.ims[0]
                    frame = annotated_frame
                except Exception as e:
                    print(f"YOLOv5 detection error: {e}")

            # Perform face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x1, y1, w, h) in faces:
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

            # Perform tracking if enabled
            if tracking and tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Tracking Failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    tracking = False

            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # Resize image to fit canvas
            img = img.resize((750, 600), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            # Keep a reference to prevent garbage collection
            image_list.append(imgtk)
            # Clear previous image
            canvas.delete("all")
            # Display on canvas
            canvas.create_image(0, 0, anchor="nw", image=imgtk)
            # Call update_frame again after delay
            root.after(30, update_frame)
        else:
            video_capture.release()
            video_capture = None
            tracking = False
            messagebox.showinfo("Video Ended", "Video playback has completed.")

# Setup GUI Layout
left_frame = tk.Frame(root, width=300, height=700, bg="white")
left_frame.pack(side="left", fill="y", padx=10, pady=10)

canvas = tk.Canvas(root, width=750, height=600, bg="grey")
canvas.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Add Buttons and Controls
image_button = tk.Button(left_frame, text="Add Image", command=add_image, bg="#4CAF50", fg="white", width=25)
image_button.pack(pady=10)

video_button = tk.Button(left_frame, text="Load Video", command=load_video, bg="#2196F3", fg="white", width=25)
video_button.pack(pady=10)

detect_tumor_button = tk.Button(left_frame, text="Detect Tumor", command=detect_tumor, bg="#f44336", fg="white", width=25)
detect_tumor_button.pack(pady=10)

start_tracking_button = tk.Button(left_frame, text="Start Tracking", command=start_tracking, bg="#FF9800", fg="white", width=25)
start_tracking_button.pack(pady=10)

color_button = tk.Button(left_frame, text="Change Pen Color", command=change_color, bg="#9C27B0", fg="white", width=25)
color_button.pack(pady=10)

# Pen Size Radio Buttons
pen_size_frame = tk.LabelFrame(left_frame, text="Pen Size", bg="white")
pen_size_frame.pack(pady=10)

pen_size_1 = tk.Radiobutton(
    pen_size_frame, text="Small", value=3, command=lambda: change_size(3), bg="white")
pen_size_1.pack(side="left", padx=5)

pen_size_2 = tk.Radiobutton(
    pen_size_frame, text="Medium", value=5, command=lambda: change_size(5), bg="white")
pen_size_2.pack(side="left", padx=5)
pen_size_2.select()

pen_size_3 = tk.Radiobutton(
    pen_size_frame, text="Large", value=7, command=lambda: change_size(7), bg="white")
pen_size_3.pack(side="left", padx=5)

# Shape Selection Radio Buttons
shape_frame = tk.LabelFrame(left_frame, text="Shapes", bg="white")
shape_frame.pack(pady=10)

line_button = tk.Radiobutton(shape_frame, text="Line", value="line", command=lambda: change_shape("line"), bg="white")
line_button.pack(side=tk.LEFT, padx=5)

oval_button = tk.Radiobutton(shape_frame, text="Oval", value="oval", command=lambda: change_shape("oval"), bg="white")
oval_button.pack(side=tk.LEFT, padx=5)

rectangle_button = tk.Radiobutton(shape_frame, text="Rectangle", value="rectangle", command=lambda: change_shape("rectangle"), bg="white")
rectangle_button.pack(side=tk.LEFT, padx=5)

text_button = tk.Radiobutton(shape_frame, text="Text", value="text", command=lambda: change_shape("text"), bg="white")
text_button.pack(side=tk.LEFT, padx=5)

# Filter Selection
filter_label = tk.Label(left_frame, text="Select Filter", bg="white")
filter_label.pack(pady=5)
filter_combobox = ttk.Combobox(left_frame, values=["Grayscale", "RGB", "Binary"], state="readonly")
filter_combobox.pack(pady=5)
filter_combobox.current(0)
filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))

filter_label2 = tk.Label(left_frame, text="Select Additional Filters", bg="white")
filter_label2.pack(pady=5)
filter_combobox1 = ttk.Combobox(left_frame, values=["Blur", "Emboss", "Sharpen", "Smooth"], state="readonly")
filter_combobox1.pack(pady=5)
filter_combobox1.current(0)
filter_combobox1.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox1.get()))

# Clear Canvas Button
clear_button = tk.Button(left_frame, text="Clear Canvas", command=clear_canvas, bg="#FF5722", fg="white", width=25)
clear_button.pack(pady=20)

# Brightness Adjustment
bright_label = tk.Label(left_frame, text="Adjust Brightness", bg="white")
bright_label.pack(pady=5)
brightness = tk.Entry(left_frame, width=20)
brightness.pack(pady=5)
brightness.insert(0, "1.0")  # Default value

submit1 = tk.Button(left_frame, text="Apply Brightness", command=lambda: adjust_brightness(brightness.get()), bg="#3F51B5", fg="white", width=25)
submit1.pack(pady=5)

# Contrast Adjustment
contrast_label = tk.Label(left_frame, text="Adjust Contrast", bg="white")
contrast_label.pack(pady=5)
contrast = tk.Entry(left_frame, width=20)
contrast.pack(pady=5)
contrast.insert(0, "1.0")  # Default value

submit2 = tk.Button(left_frame, text="Apply Contrast", command=lambda: adjust_contrast(contrast.get()), bg="#009688", fg="white", width=25)
submit2.pack(pady=5)

# Bind Canvas Events for Drawing
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<ButtonRelease-1>", end_draw)

# Start Tkinter Main Loop
root.mainloop()
