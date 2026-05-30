import sys
print("Starting imports...")
try:
    import cv2
    print("cv2 imported")
    import torch
    print("torch imported")
    import ultralytics
    print("ultralytics imported")
    import mediapipe
    print("mediapipe imported")
except Exception as e:
    print(f"Error importing: {e}")

print("Imports done.")
