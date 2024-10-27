import time
import os
from datetime import datetime

def create():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.txt"
    dest_path = "C:/Users/Sridhar/OneDrive/Desktop/21L105/sem7/DL/Assignment/results/"
    
    # Ensure the directory exists
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)  # Create the directory if it doesn't exist
    
    destination = os.path.join(dest_path, filename)
    
    # Create the file with an initial line
    if not os.path.exists(destination):
        with open(destination, "w") as file:
            file.write("This file was created with a timestamp.\n")

    return destination

def writer(val, destination):
    with open(destination, "a") as log_file:
        log_file.write(f"\n{val}")

