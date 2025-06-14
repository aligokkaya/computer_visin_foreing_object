import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2

# Veri klasörü
crop_dir = "dataset/train/clean_crops_320"
X = []

# Veriyi yükle
for fname in os.listdir(crop_dir):
    img_path = os.path.join(crop_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (320, 320))
        img = img.astype("float32") / 255.0
        X.append(img)

X = np.array(X)
X = np.expand_dims(X, axis=-1)  # (N, 320, 320, 1)

X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

input_img = Input(shape=(320, 320, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("models/autoencoder_320_best_v2.h5", monitor="val_loss", save_best_only=True)

autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=16,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stop, checkpoint]
)