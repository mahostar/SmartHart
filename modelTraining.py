import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import joblib

# Enable GPU memory growth to prevent TF from taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_and_preprocess_data(file_path, seq_length=10, overlap=0.0):
    """
    1) Loads CSV data from 'file_path'.
    2) Parses timestamps with infer_datetime_format=True (handles both 
       fractional and non-fractional seconds).
    3) Sorts by timestamp.
    4) Scales numeric features.
    5) Encodes anomaly labels (normal, septic_shock, etc.) -> integers.
    6) Converts to overlapping sequences of length seq_length.
    """

    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    # 1) Read data
    df = pd.read_csv(file_path)

    # Ensure the expected columns exist
    required_cols = [
        'timestamp', 
        'systolic_bp', 
        'diastolic_bp', 
        'heart_rate', 
        'spo2', 
        'respiratory_rate', 
        'anomaly_type'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 2) Parse timestamps (some have fractional seconds, some don't)
    df['timestamp'] = pd.to_datetime(
        df['timestamp'],
        infer_datetime_format=True,  # Let pandas guess the correct format
        errors='coerce'             # Non-parsable => NaT
    )
    # Drop rows that failed to parse or have NaT
    df.dropna(subset=['timestamp'], inplace=True)

    # Sort by timestamp in ascending order
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Dataframe shape after timestamp parse: {df.shape}")

    # 3) Basic features + optional derived features
    feature_cols = [
        'systolic_bp', 
        'diastolic_bp', 
        'heart_rate', 
        'spo2',
        'respiratory_rate'
    ]
    # Example derived features
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['shock_index'] = df['heart_rate'] / (df['systolic_bp'] + 1e-6)

    # Extend the feature list
    derived_cols = ['pulse_pressure', 'shock_index']
    all_features = feature_cols + derived_cols

    # Drop any rows with NaN in these columns
    df.dropna(subset=all_features, inplace=True)

    # 4) Scale features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[all_features])
    scaled_df = pd.DataFrame(scaled_array, columns=all_features)

    # 5) Label Encode anomaly_type
    label_encoder = LabelEncoder()
    df['anomaly_type'].fillna('normal', inplace=True)
    labels = label_encoder.fit_transform(df['anomaly_type'])

    # 6) Convert to overlapping sequences
    step = int(seq_length * (1 - overlap))
    step = max(step, 1)  # ensure we don't get stuck with 0

    X, y = [], []
    for start_idx in range(0, len(scaled_df) - seq_length + 1, step):
        end_idx = start_idx + seq_length
        seq_x = scaled_df.iloc[start_idx:end_idx].values
        seq_y = labels[end_idx - 1]  # label from the last row
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    print(f"Number of sequences created: {len(X)}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    return X, y, label_encoder, scaler, all_features

def build_model(seq_length, num_features, num_classes):
    """
    LSTM or BiLSTM classification model for anomaly detection.
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), 
                      input_shape=(seq_length, num_features)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_curves(history):
    """
    Show training vs validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.grid(True)
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """
    Plot confusion matrix with class labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_roc_curves(y_true, y_pred_proba, label_encoder):
    """
    Plot multi-class ROC curves for each label.
    """
    n_classes = len(label_encoder.classes_)
    y_true_bin = to_categorical(y_true, n_classes)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC={score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def main():
    # Model Configuration
    CSV_FILE_PATH = r"C:\Users\Mohamed\Desktop\proj\health_monitoring_data\health_data.csv"
    SEQ_LENGTH = 10          # Sequence length of 10 time steps
    OVERLAP = 0.0           # Overlap ratio between sequences
    TEST_SIZE = 0.2         # Train/test split ratio
    VAL_SPLIT = 0.2         # Validation split ratio
    EPOCHS = 30             # Maximum number of training epochs
    BATCH_SIZE = 32         # Training batch size
    RANDOM_SEED = 42        # Random seed for reproducibility

    # 1) Load and Preprocess Data
    X, y, label_encoder, scaler, feature_names = load_and_preprocess_data(
        file_path=CSV_FILE_PATH,
        seq_length=SEQ_LENGTH,
        overlap=OVERLAP
    )
    
    # 2) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        shuffle=False,  # Keep chronological order
        random_state=RANDOM_SEED
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 3) Build Model
    num_features = X.shape[2]
    num_classes = len(np.unique(y))
    model = build_model(SEQ_LENGTH, num_features, num_classes)
    model.summary()

    # 4) Train Model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 5) Evaluate Model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # 6) Visualize Results
    # Training curves
    plot_training_curves(history)

    # Predictions and metrics
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_pred_proba = model.predict(X_test)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # ROC curves
    plot_roc_curves(y_test, y_pred_proba, label_encoder)

    # 7) Save Model & Preprocessing Objects
    print("\nSaving model and preprocessing objects...")
    model.save("health_monitor_model.h5")
    np.save("label_encoder_classes.npy", label_encoder.classes_)
    joblib.dump(scaler, "scaler.pkl")
    print("All done!")

if __name__ == "__main__":
    main()