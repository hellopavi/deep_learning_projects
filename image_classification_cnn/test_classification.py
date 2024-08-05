from tensorflow.keras.preprocessing import image
from keras.models import load_model , model_from_json
import numpy as np
import os

# Load the model architecture from the JSON file
with open('model.json' , 'r', encoding='utf-8') as json_file :
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model weights from the HDF5 file
model.load_weights("model.weights.h5")
print("Model Loaded")

def classify(img_file):
    try :
        # Load and preprocess the image
        img_name = img_file
        test_image = image.load_img(img_name, target_size=(256, 256), color_mode = 'grayscale')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Predict the class of the image
        result = model.predict(test_image)
        
        # Process the result to get the class with the highest probability
        arr = np.array(result[0])
        print("ARRAY", arr)
        maxx = np.amax(arr)
        max_prob = arr.argmax(axis=0)
        max_prob = max_prob + 1
        
        # Define the classes
        classes = ["zero", "one", "two", "three", "four", "five"]
        
        # Get the resulting class based on the prediction
        result = classes[max_prob - 1]
        
        # Print the result
        print("Img name:", img_name, "RESULT:", result)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

# Define the path to the directory containing images to classify
path = 'dataset/test'

# Create a list to hold the file paths of all images in the directory
files = []

# Traverse the directory and collect all image files with .jpg extension
# r = root , d = directory , f = files
for r , d , f in os.walk(path):
    for file in f :
        if '.jpg' in file :
            files.append(os.path.join(r,file))
            
# Classify each image file found
for f in files :
    classify(f)

print("Succesfully predicted . . .")
