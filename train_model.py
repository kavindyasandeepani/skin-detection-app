import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Images Load 
# ============================================
print("📂 Loading dataset...")

train_data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = train_data.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_data.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("\n✅ Classes Found:", train_generator.class_indices)

# ============================================
# STEP 2: CNN Model Structure
# ============================================
print("\n🏗️  Building Model...")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')   # 4 = acne, dry, oily, normal
])

# ============================================
# STEP 3: Model Compile 
# ============================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 4: Model Train 
# ============================================
print("\n🚀 Training Started...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# ============================================
# STEP 5: Model Save 
# ============================================
model.save('skin_model.h5')
print("\n✅ Training Complete! Model saved as 'skin_model.h5'")

# ============================================
# STEP 6: Training Graph 
# ============================================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_graph.png')
print("📊 Training graph saved as 'training_graph.png'")