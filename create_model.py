import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Cast
from tensorflow.keras.models import Model
import os

def create_model():
    # Create the base VGG19 model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Save the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breast_cancer_vgg19.h5')
    model.save(model_path)
    print(f"Model created and saved successfully at {model_path}")

if __name__ == "__main__":
    create_model() 