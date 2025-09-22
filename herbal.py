import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import os

class HerbalPlantDetector:
    def __init__(self):
        self.class_names = ['Tulsi', 'Neem', 'Aloe Vera', 'Mint']  # Add more herb classes as needed
        self.img_size = (224, 224)
        self.model = self.create_model()

    def create_model(self):
        # Creating a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_image(self, img):
        # Preprocess the image for the model
        img = cv2.resize(img, self.img_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def detect_from_file(self, image_path):
        # Read the image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read and process the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Preprocess the image
        processed_img = self.preprocess_image(img)
        
        # Get prediction
        predictions = self.model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        return img, self.class_names[predicted_class], confidence

def main():
    # Initialize detector
    detector = HerbalPlantDetector()
    
    # Get image path from user
    image_path = input("Enter the path to your plant image: ")
    
    try:
        # Detect plant from image
        img, plant_name, confidence = detector.detect_from_file(image_path)
        
        # Draw results on image
        display_img = img.copy()
        cv2.putText(display_img, f'Plant: {plant_name}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, f'Confidence: {confidence:.2f}%', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Herbal Plant Detection', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print results
        print(f"\nResults:")
        print(f"Detected Plant: {plant_name}")
        print(f"Confidence: {confidence:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()