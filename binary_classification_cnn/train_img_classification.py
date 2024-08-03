import tensorflow  # Import TensorFlow for deep learning operations
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the convolutional neural network (CNN) model architecture
model = Sequential([
    layers.Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(units=128 , activation='relu'),
    layers.Dense(units=1 , activation='sigmoid')])


# Compile the model, specifying optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics =['accuracy'])


# Create data augmentation generators for (training and validation data)
train_datagen = ImageDataGenerator(
    rescale=1./255 ,
    shear_range = 0.2,
    zoom_range= 0.2 ,
    horizontal_flip= True )

val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training and validation datasets from directories
training_set = train_datagen.flow_from_directory(
    'dataset/train' ,
    target_size = (64,64) ,
    batch_size=10 ,
    class_mode='binary') 

val_set = val_datagen.flow_from_directory(
    'dataset/valid' ,
    target_size = (64,64) ,
    batch_size=10 ,
    class_mode='binary') 

# Train the model on the training data with validation
model.fit(training_set ,
          steps_per_epoch=10 ,
          epochs=50 ,
          validation_data= val_set )

# Function to save the model weights and architecture
def save() :
    model_json = model.to_json()
    with open('model.json' , 'w') as json_file :
        json_file.write(model_json)
    model.save_weights("model.weights.h5")
    

# Get user input in lowercase for saving the model
save1 = str(input('Do you want to save the model Yes or No : ')).lower()
print(save1)

if save1 == 'yes' :
    save()
    print('Model saved')
else :
    print('Model not saved')
