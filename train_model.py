import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Paths to the dataset
REAL_PATH = 'dataset/real'
FAKE_PATH = 'dataset/fake'

# Image dimensions (make sure this matches your extracted frame size)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load images and labels
data = []
labels = []

print("🔄 Loading real images...")
for img_name in os.listdir(REAL_PATH):
    img_path = os.path.join(REAL_PATH, img_name)
    try:
        image = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image = img_to_array(image)
        data.append(image)
        labels.append(0)  # Real label = 0
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

print("🔄 Loading fake images...")
for img_name in os.listdir(FAKE_PATH):
    img_path = os.path.join(FAKE_PATH, img_name)
    try:
        image = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image = img_to_array(image)
        data.append(image)
        labels.append(1)  # Fake label = 1
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split into training and testing
print("📚 Splitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
print("🛠️ Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training the model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
os.makedirs('models', exist_ok=True)
model.save('models/deepfake_detector_model.h5')

print("✅ Model training completed and saved in 'models/deepfake_detector_model.h5'!")
