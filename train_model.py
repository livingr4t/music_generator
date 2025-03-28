import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Wczytaj dane treningowe
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Zakodowanie y_train w formacie one-hot
num_classes = 128  # Liczba unikalnych nut (MIDI ma zakres 0-127)
y_train = to_categorical(y_train, num_classes=num_classes)

# Dopasowanie wymiarów X_train
# Upewnij się, że dane mają odpowiedni kształt (num_samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Budowa modelu LSTM
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Dopasowanie liczby klas
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trening modelu
model.fit(X_train, y_train, epochs=20, batch_size=64)

# Zapisanie modelu
model.save("music_model.h5")
print("Model został zapisany jako 'music_model.h5'")
