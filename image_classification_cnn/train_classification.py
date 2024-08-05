import tensorflow
from tensorflow.keras import layers  
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

# Define the CNN model architecture
model = Sequential([
    # First convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and input shape
    layers.Conv2D(32, (3,3) , input_shape=(256,256,1), activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    # Second convolutional layer with 64 filters and ReLU activation
    layers.Conv2D(64, (3,3) ,  activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    
    # Third convolutional layer with 128 filters and ReLU activation
    layers.Conv2D(128, (3,3) ,  activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    # Fourth convolutional layer with 256 filters and ReLU activation
    layers.Conv2D(256, (3,3) , activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),

    # Flatten the output from the convolutional layers to feed into the fully connected layers
    layers.Flatten(),

    # Fully connected layer with 128 units and ReLU activation
    layers.Dense(units=128 , activation='relu'),
    layers.Dropout(0.25),

    # Output layer with 6 units (one for each class) and softmax activation
    layers.Dense(units=6  , activation='softmax')
])

# Compile the model
model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])

# Create an ImageDataGenerator for the dataset
train_datagen = ImageDataGenerator(rescale=1./255 , rotation_range=12. , width_shift_range=0.2, height_shift_range=0.2 ,zoom_range= 0.2 , horizontal_flip= True )

val_datagen = ImageDataGenerator(rescale= 1./255)

# Load the training data from the director
training_set = train_datagen.flow_from_directory('dataset/train' , target_size = (256,256) , color_mode='grayscale' , classes=['0','1' , '2', '3' , '4' , '5'], batch_size=10 , class_mode='categorical') 
print(training_set.classes.shape)
val_set = val_datagen.flow_from_directory('dataset/validation' , target_size = (256,256) , color_mode='grayscale' , classes=['0','1' , '2', '3' , '4' , '5'],batch_size=10 , class_mode='categorical') 

# Define callbacks for early stopping and model checkpoint
callback_list = [
    EarlyStopping(monitor='val_loss' , patience=10 ),
    ModelCheckpoint(filepath="weights.h5.keras", monitor='val_loss', save_best_only=True,verbose=1)
]

# Train the model
model.fit(training_set , steps_per_epoch=30 ,epochs=30 , validation_data= val_set , callbacks = callback_list )

# Save the model architecture to a JSON file
model_json = model.to_json()
with open('model.json' , 'w') as json_file :
    json_file.write(model_json)

# Save the model weights to a HDF5 file
model.save_weights("model.weights.h5")
print("Model saved")
