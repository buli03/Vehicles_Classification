# Vehicle Image Classification using CNN

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify vehicle images into 7 categories. The model achieves 83% validation accuracy using TensorFlow/Keras. The dataset contains 5,587 images across the following classes:
- Auto Rickshaws
- Bikes
- Cars
- Motorcycles
- Planes
- Ships
- Trains

## Key Features
- Automated dataset download from Google Drive
- Image preprocessing pipeline
- CNN architecture with 3 convolutional layers
- Training/validation split (80/20)
- Model performance visualization
- Detailed classification metrics

# Usage

## Download dataset 
```
!pip install gdown
!gdown 1MWMTJt9fn-62pN7mFhKQKKZRlX4gh4UF -O veh_img.zip
!unzip -q veh_img.zip
```
## Train model 
```
# Create datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
  'veh_img',
  validation_split=0.2,
  subset='training',
  seed=1,
  image_size=(64, 64),
  batch_size=32)

# Build model
model = Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(7)
])

# Train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=20)
```

## Evaluate model
```
evaluate_model(val_ds, model)  # Generates classification report and confusion matrix
```

# Dataset Details
### Source: Kaggle Vehicle Classification Dataset: https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification
### Total Images: 5,587
### Class Distribution:
- Auto Rickshaws: 800
- Bikes: 800
- Cars: 790
- Motorcycles: 800
- Planes: 800
- Ships: 797
- Trains: 800

## Model architecture
```
Input: 64x64 RGB images
-------------------------------------------------
Layer (type)                 Output Shape      
=================================================
Rescaling (Rescaling)        (None, 64, 64, 3)  
Conv2D (16 filters)          (None, 64, 64, 16) 
MaxPooling2D                 (None, 32, 32, 16) 
Conv2D (32 filters)          (None, 32, 32, 32) 
MaxPooling2D                 (None, 16, 16, 32) 
Conv2D (64 filters)          (None, 16, 16, 64) 
MaxPooling2D                 (None, 8, 8, 64)   
Flatten                      (None, 4096)       
Dense (128 units)            (None, 128)        
Output (7 units)             (None, 7)          
```

# Example results
```
               Precision  Recall  F1-Score
Auto Rickshaws    0.80     0.84      0.82
Bikes             0.93     0.92      0.92
Cars              0.77     0.75      0.76
Motorcycles       0.78     0.82      0.80
Planes            0.83     0.83      0.83
Ships             0.84     0.82      0.83
Trains            0.84     0.81      0.83

Accuracy: 0.83
```
![image](https://github.com/user-attachments/assets/6ab4f7ae-4b22-4e87-accf-755c3ba8b24e)
