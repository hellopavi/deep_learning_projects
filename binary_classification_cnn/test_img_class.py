import tensorflow.keras.models as models
import numpy as np
from keras.preprocessing import image 
import os

# Load the model architecture from a JSON file
with open('model.json' , 'r') as json_file :
    loaded_model_json = json_file.read()
json_file.close()
model = models.model_from_json(loaded_model_json)
# Load the model weights from a HDF5 file
model.load_weights('model.weights.h5')
print('Model loaded')


# Function to classify an image
def classify_img(img_file) :
    img_name = img_file

    # Load and preprocess the image
    test_img = image.load_img(img_name , target_size=(64,64))  
    test_img= image.img_to_array((test_img))
    test_img = np.expand_dims(test_img , axis=0)

    # Make prediction using the loaded model
    result = model.predict(test_img)

    # Determine the predicted class based on the output
    if result[0][0] == 1 :
        prediction = 'Tony stark'
    else :
        prediction = 'Elon musk'

    # Print the prediction and image name
    print(f'Image input : {img_name}\n Output : {prediction}')


# Define the path to the test image directory
path = 'dataset_structure/test'
files = []

# Find all JPEG images in the test directory
# r=root , d = directories , f = files
for r , d , f in os.walk(path):
    for file in f:
        if '.jpeg' in file :
            files.append(os.path.join(r, file))
# Classify each image in the test set
for f in files :
    classify_img(f)
