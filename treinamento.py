import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Função para aplicar aumentos no áudio
def augment_audio(y, sr):
    # Alteração de pitch
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    # Alteração de tempo (speed up/down)
    y_speed = librosa.effects.time_stretch(y, rate=1.2)
    # Adicionar ruído
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    return y_pitch, y_speed, y_noisy

# Função para padronizar o comprimento dos espectrogramas
def pad_spectrogram(spectrogram, max_length):
    if spectrogram.shape[1] > max_length:
        return spectrogram[:, :max_length]
    else:
        padding = max_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')

# Função para normalizar o espectrograma
def normalize_spectrogram(spectrogram):
    return (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

# Função para carregar e processar o dataset
def load_data(data_dir, max_length=128):
    labels = []
    features = []
    label_map = {'ambulance': 0, 'dog': 1, 'firetruck': 2, 'traffic': 3}
    
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for audio_file in os.listdir(category_dir):
                file_path = os.path.join(category_dir, audio_file)
                y, sr = librosa.load(file_path, sr=None)
                
                # Aplicar aumentos
                y_pitch, y_speed, y_noisy = augment_audio(y, sr)
                
                for audio in [y, y_pitch, y_speed, y_noisy]:
                    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                    
                    # Normalizar o espectrograma
                    spectrogram = normalize_spectrogram(spectrogram)
                    
                    # Padronizar o tamanho do espectrograma
                    spectrogram = pad_spectrogram(spectrogram, max_length)
                    
                    features.append(spectrogram)
                    labels.append(label_map[category])
    
    # Convertendo listas para arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Ajuste de dimensões para o modelo convolucional
    features = features[..., np.newaxis]
    
    return features, labels

# Função para criar o modelo CNN com dropout e regularização L2
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer='l2', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Adiciona dropout para evitar overfitting
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dense(4, activation='softmax')  # Ajustar para 4 classes
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Função para treinar o modelo usando validação cruzada
def train_model_with_cross_validation(features, labels):
    skf = StratifiedKFold(n_splits=3)
    fold_no = 1
    
    for train_idx, val_idx in skf.split(features, labels):
        print(f'Treinando o Fold {fold_no}...')
        train_X, val_X = features[train_idx], features[val_idx]
        train_y, val_y = labels[train_idx], labels[val_idx]

        model = create_cnn_model(input_shape=(features.shape[1], features.shape[2], 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            ModelCheckpoint(f'model_fold_{fold_no}.h5', save_best_only=True, save_weights_only=True)
        ]
        
        history = model.fit(train_X, train_y, epochs=15, batch_size=32, validation_data=(val_X, val_y), callbacks=callbacks)

        # Gráfico de desempenho por fold
        plt.figure()
        plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
        plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
        plt.title(f'Fold {fold_no} - Acurácia')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.show()
        
        fold_no += 1

    # Salvar o modelo final
    model.save('modelo_final.h5')
    print("Modelo salvo como 'modelo_final.h5'.")

# Executando o treinamento
features, labels = load_data('dataset')
train_model_with_cross_validation(features, labels)
