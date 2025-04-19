import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Dropout,
                                     MaxPooling1D, Bidirectional, LSTM, Dense,
                                     LayerNormalization, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K
import tensorflow as tf

# ------------------------------
# 1. Load & Prepare Data
# ------------------------------
df = pd.read_csv("features_wav2vec2.csv")
le = joblib.load("label_encoder.pkl")
num_classes = len(le.classes_)

X = df.drop(columns=["emotion", "dataset_name"]).values
y = df["emotion"].values
y_encoded = to_categorical(y, num_classes)

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ------------------------------
# 2. Focal Loss Function
# ------------------------------
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss

# ------------------------------
# 3. Build Model
# ------------------------------
input_layer = Input(shape=(X_train.shape[1], 1))

x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LayerNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = LayerNormalization()(x)
x = Dropout(0.4)(x)

#  Add Attention layer
attention = Attention()([x, x])
x = tf.keras.layers.GlobalAveragePooling1D()(attention)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# ------------------------------
# 4. Compile
# ------------------------------
initial_lr = 0.0005
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=initial_lr,
    first_decay_steps=10,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-6
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
    metrics=["accuracy", Precision(name='precision'), Recall(name='recall')]
)

# ------------------------------
# 5. Callbacks & Training
# ------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint("best_emotion_model.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------
# 6. Evaluation & Save
# ------------------------------
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + K.epsilon())

print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {f1_score:.4f}")

model.save("final_emotion_model.keras")
