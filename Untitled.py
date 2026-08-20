!mkdir -p safespace_fusion/models


# Create core files
open('safespace_fusion/fusion_engine.py', 'w').close()
open('safespace_fusion/main.py', 'w').close()

# Create model files
open('safespace_fusion/models/biomedical_model.py', 'w').close()
open('safespace_fusion/models/behavioral_model.py', 'w').close()
open('safespace_fusion/models/text_model.py', 'w').close()
open('safespace_fusion/models/voice_model.py', 'w').close()
open('safespace_fusion/models/facial_model.py', 'w').close()



!tree safespace_fusion


import os

# Walk through the safespace_fusion directory and print all files/folders
for root, dirs, files in os.walk("safespace_fusion"):
    level = root.replace("safespace_fusion", "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = " " * 4 * (level + 1)
    for f in files:
        print(f"{sub_indent}{f}")



!python3 safespace_fusion/main.py


!pip install transformers



import os
print(os.getcwd())


pip install timm opencv-python torch torchvision



!python safespace_fusion/main.py


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

def safe_divide(a, b):
    """Safe division with zero handling"""
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def load_components():
    """Load all components with validation"""
    try:
        # Load ELM weights
        elm_weights = np.load('models/elm_weights.npz')
        W = elm_weights['W']
        b = elm_weights['b']
        B = elm_weights.get('beta', elm_weights.get('B'))  # Try both keys
        
        if B is None:
            raise ValueError("Couldn't find output weights in elm_weights.npz")
        
        # Load and validate scalers
        scaler_mean = np.load('features/scaler_mean.npy')
        scaler_scale = np.load('features/scaler_scale.npy')
        scaler_scale[scaler_scale == 0] = 1.0  # Prevent division by zero
        
        # Load models
        #resnet_model = tf.saved_model.load('models/resnet50_features')
        #vit_model = tf.saved_model.load('models/vit_features')
        # NEW (Correct for Keras SavedModels)
        resnet_model = tf.keras.models.load_model('models/resnet50_features')
        vit_model = tf.keras.models.load_model('models/vit_features')
        # Verify model signatures
        if 'serving_default' not in resnet_model.signatures:
            raise ValueError("ResNet model missing serving_default signature")
        if 'serving_default' not in vit_model.signatures:
            raise ValueError("ViT model missing serving_default signature")
        
        return {
            'W': W, 'b': b, 'B': B,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
            'resnet_model': resnet_model,
            'vit_model': vit_model,
            'face_cascade': cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        }
    
    except Exception as e:
        raise RuntimeError(f"Loading failed: {str(e)}")

def extract_features(face_img, models):
    """Robust feature extraction"""
    try:
        img = cv2.resize(face_img, (224, 224))
        img_tensor = tf.convert_to_tensor(
            np.expand_dims(image.img_to_array(img)/255.0, axis=0),
            dtype=tf.float32
        )
        
        # Get features from both models
        resnet_out = models['resnet_model'].signatures['serving_default'](img_tensor)
        vit_out = models['vit_model'].signatures['serving_default'](img_tensor)
        
        # Convert outputs safely
        resnet_features = next(iter(resnet_out.values())).numpy().flatten()
        vit_features = next(iter(vit_out.values())).numpy().flatten()
        
        return np.concatenate([resnet_features, vit_features])
    
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def get_stress_score(image_path):
    """Main prediction function"""
    try:
        # Load with validation
        components = load_components()
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Invalid image file"
            
        # Detect face
        faces = components['face_cascade'].detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
            scaleFactor=1.1, 
            minNeighbors=5
        )
        if len(faces) == 0:
            return "Error: No face detected"
        
        # Process first face
        x,y,w,h = faces[0]
        features = extract_features(img[y:y+h, x:x+w], components)
        
        # Safe normalization
        features_norm = safe_divide(
            features - components['scaler_mean'],
            components['scaler_scale']
        )
        
        # Stable calculation
        H = np.maximum(0.01 * (features_norm @ components['W'] + components['b']), 
                      features_norm @ components['W'] + components['b'])
        stress_score = (H @ components['B']).item()
        
        return f"{np.clip(stress_score, 0, 1):.4f}"  # Clipped to valid range
    
    except Exception as e:
        return f"Error: {str(e)}"

if _name_ == "_main_":
    try:
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            print(get_stress_score(file_path))
    except Exception as e:
        print(f"Fatal error: {str(e)}")


!pip install opencv-python



import websocket
import json
import time
import numpy as np

# Store recent BVP values with timestamps
bvp_data = []

def calculate_bpm(bvp_data):
    if len(bvp_data) < 2:
        return 0

    values = [val for (val, ts) in bvp_data]
    timestamps = [ts for (val, ts) in bvp_data]

    peaks = []
    threshold = np.mean(values)

    for i in range(1, len(values) - 1):
        if values[i] > threshold and values[i] > values[i-1] and values[i] > values[i+1]:
            peaks.append(timestamps[i])

    if len(peaks) < 2:
        return 0

    rr_intervals = np.diff(peaks)
    bpm = 60 / np.mean(rr_intervals)
    return bpm

def estimate_stress(bpm):
    if bpm < 60:
        return "Low"
    elif bpm < 90:
        return "Medium"
    else:
        return "High"

def on_message(ws, message):
    global bvp_data

    try:
        data = json.loads(message)
        bvp = int(data["BVP"])
        timestamp = time.time()
        bvp_data.append((bvp, timestamp))

        bvp_data = [(v, t) for v, t in bvp_data if timestamp - t <= 10]

        if len(bvp_data) >= 30:
            bpm = calculate_bpm(bvp_data)
            stress = estimate_stress(bpm)
            print(f"BPM: {bpm:.2f} | Stress: {stress}")

    except Exception as e:
        print("WebSocket error:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://172.20.10.3:81",  # <-- Updated IP here
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


import websocket
import json

def on_message(ws, message):
    try:
        data = json.loads(message)
        print(f"Received: {data}")
    except Exception as e:
        print("Error parsing message:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://172.16.90.91:81",  # Update if your ESP32 IP changes
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


import serial
import json
from time import sleep

# === Configuration ===
SERIAL_PORT = '/dev/tty.usbserial-0001'   # ← Change to your correct port
BAUD_RATE = 115200

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")

        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Store values into variables
                bvp     = data.get('BVP', 0)
                gsr     = data.get('GSR', 0)
                temp    = data.get('Temp', 0.0)
                accel_x = data.get('AccelX', 0.0)
                accel_y = data.get('AccelY', 0.0)
                accel_z = data.get('AccelZ', 0.0)

                # Print or use the variables as needed
                print(f"BVP: {bvp}, GSR: {gsr}, Temp: {temp}, Accel: ({accel_x}, {accel_y}, {accel_z})")

                # === Example: Simple stress detection logic ===
                # (Replace this with your actual model/rules)
                if gsr > 2500 and temp > 37.5:
                    print("⚠️  Possible stress detected!")

            except json.JSONDecodeError:
                print("Invalid JSON:", line)

            # Optional delay (e.g., to reduce CPU usage)
            sleep(0.01)

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")

if __name__ == "__main__":
    main()



import serial
import json
import time

# === CONFIG ===
SERIAL_PORT = "/dev/tty.usbserial-0001"  # ← Update if needed (you already confirmed this is correct)
BAUD_RATE = 115200

# === Variables to hold latest sensor values ===
latest_data = {
    "BVP": 0,
    "GSR": 0,
    "Temp": 0.0,
    "AccelX": 0.0,
    "AccelY": 0.0,
    "AccelZ": 0.0
}

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    time.sleep(2)  # Wait for ESP32 to finish reset

    while True:
        line = ser.readline().decode("utf-8").strip()
        if line:
            try:
                data = json.loads(line)
                latest_data.update(data)

                # === Print or use the values ===
                print(f"BVP: {latest_data['BVP']}, GSR: {latest_data['GSR']}, Temp: {latest_data['Temp']} °C")
                print(f"Accel -> X: {latest_data['AccelX']}, Y: {latest_data['AccelY']}, Z: {latest_data['AccelZ']}")
                print("-" * 50)

                # You can now use latest_data['BVP'], etc. for your stress detection

            except json.JSONDecodeError:
                print("Invalid JSON:", line)

except serial.SerialException as e:
    print("Serial connection error:", e)



import serial
import json

# Open serial port
ser = serial.Serial('/dev/tty.usbserial-0001', 115200)
latest_data = {}

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        
        # Ensure the line is a valid JSON
        if not line.startswith("{") or not line.endswith("}"):
            continue  # skip partial/bad lines
        
        data = json.loads(line)
        latest_data = data
        
        # Optional print for debug
        print(f"BVP: {data['BVP']}, GSR: {data['GSR']}, Temp: {data['Temp']} °C")
        print(f"Accel -> X: {data['AccelX']}, Y: {data['AccelY']}, Z: {data['AccelZ']}")
        print("-" * 50)

    except json.JSONDecodeError:
        print(f"Invalid JSON: {line}")
    except Exception as e:
        print(f"Error: {e}")




