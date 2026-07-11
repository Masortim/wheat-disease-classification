import os
import sys
import gc
import warnings
import json
import glob
import shutil
import time
from datetime import datetime

# =============================================================================
# GPU MEMORY CONFIGURATION — must be set BEFORE importing TensorFlow
# =============================================================================
# Use CUDA async allocator to prevent memory fragmentation over long training
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, roc_curve, auc, f1_score, recall_score)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, 
                                     Dropout, BatchNormalization,
                                     Activation, Input, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                        ReduceLROnPlateau, Callback)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# Configure GPU memory growth to prevent OOM on long training runs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[*] Настроен GPU memory growth для {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[!] Ошибка настройки GPU: {e}")

# Включаем смешанную точность для ускорения (если GPU поддерживает)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

warnings.filterwarnings('ignore')

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
class Config:
    DATASET_ROOT = r'D:\dataset\wheat_desease'
    OUTPUTS_DIR = 'outputs'
    CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
    REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')
    MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
    
    # FIX: Reduced from 255x255 to 224x224 (standard size, less memory)
    IMG_SIZE = (224, 224)
    # FIX: Reduced from 64 to 32 (prevents OOM with large feature maps)
    BATCH_SIZE = 32
    EPOCHS = 200                        # максимальное количество эпох (страховка)
    INITIAL_LR = 0.001
    MIN_LR = 1e-7
    TARGET_RECALL_PER_CLASS = 0.95      # целевая точность по классам (recall)
    
    AUGMENTATION = {
        'rescale': 1./255,
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    }


# =============================================================================
# ПРОГРЕСС-БАР
# =============================================================================
class ProgressBar:
    def __init__(self, total, desc="Processing", bar_len=40):
        self.total = total
        self.desc = desc
        self.bar_len = bar_len
        self.current = 0
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        filled = int(self.bar_len * percent)
        bar = '█' * filled + '░' * (self.bar_len - filled)
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        sys.stdout.write(f'\r{self.desc} |{bar}| {self.current}/{self.total} ({percent*100:.1f}%) {eta_str}')
        sys.stdout.flush()
        
    def close(self):
        sys.stdout.write('\n')
        sys.stdout.flush()


# =============================================================================
# УТИЛИТЫ
# =============================================================================
def ensure_dirs():
    for d in [Config.CHECKPOINTS_DIR, Config.REPORTS_DIR, Config.MODELS_DIR]:
        os.makedirs(d, exist_ok=True)


def get_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(Config.CHECKPOINTS_DIR, 'epoch_*.h5'))
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    latest = checkpoints[-1]
    epoch = int(os.path.basename(latest).split('_')[1].split('.')[0])
    return latest, epoch


def cleanup_old_checkpoints(current_epoch):
    checkpoints = glob.glob(os.path.join(Config.CHECKPOINTS_DIR, 'epoch_*.h5'))
    for ckpt in checkpoints:
        epoch = int(os.path.basename(ckpt).split('_')[1].split('.')[0])
        if epoch != current_epoch:
            try:
                os.remove(ckpt)
            except Exception:
                pass


# =============================================================================
# КАСТОМНЫЙ CALLBACK ДЛЯ ВЫЧИСЛЕНИЯ F1 НА ВАЛИДАЦИИ (в конце эпохи)
# =============================================================================
class F1ValidationCallback(Callback):
    """Вычисляет F1-score (weighted) на валидационных данных в конце каждой эпохи."""
    def __init__(self, validation_generator, num_classes, class_names=None):
        super().__init__()
        self.validation_generator = validation_generator
        self.num_classes = num_classes
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Получаем все предсказания и истинные метки из генератора
        self.validation_generator.reset()
        y_true = []
        y_pred = []
        steps = len(self.validation_generator)
        for _ in range(steps):
            x, y = self.validation_generator.next()
            preds = self.model.predict(x, verbose=0)
            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        # Обрезаем до точного количества образцов (на случай, если последний батч неполный)
        total_samples = self.validation_generator.samples
        y_true = np.array(y_true[:total_samples])
        y_pred = np.array(y_pred[:total_samples])
        # Вычисляем weighted F1
        f1 = f1_score(y_true, y_pred, average='weighted')
        logs['val_f1'] = f1
        print(f" - val_f1: {f1:.4f}")   # печатается рядом с выводом эпохи


# =============================================================================
# ОСТАЛЬНЫЕ КОЛБЭКИ
# =============================================================================
class CheckpointCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        cleanup_old_checkpoints(epoch + 1)


# FIX: Added memory cleanup callback to prevent OOM on long training runs
class MemoryCleanupCallback(Callback):
    """Очищает память GPU между эпохами для предотвращения OOM."""
    def on_epoch_end(self, epoch, logs=None):
        # Clear Keras session cache
        K.clear_session()
        # Force garbage collection
        gc.collect()


class TrainingLogger(Callback):
    def __init__(self):
        super().__init__()
        self.history_data = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        entry = {'epoch': epoch + 1, 'timestamp': datetime.now().isoformat()}
        entry.update({k: float(v) for k, v in logs.items()})
        self.history_data.append(entry)
        
    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history_data, f, indent=2, ensure_ascii=False)


# =============================================================================
# ПОСТРОЕНИЕ МОДЕЛИ
# =============================================================================
def build_model(num_classes, input_shape=(224, 224, 3)):
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Block 4
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Block 5
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Classifier
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))
    ])
    
    optimizer = Adam(learning_rate=Config.INITIAL_LR)
    # Компилируем БЕЗ кастомной метрики F1 – только стандартные метрики
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model


# =============================================================================
# ВЫЧИСЛЕНИЕ ВЕСОВ КЛАССОВ
# =============================================================================
def compute_class_weights(generator):
    classes = generator.classes
    class_indices = generator.class_indices
    num_classes = len(class_indices)
    
    unique, counts = np.unique(classes, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    total = len(classes)
    weights = {}
    
    for cls_idx in range(num_classes):
        count = count_dict.get(cls_idx, 1)
        base_weight = total / (num_classes * count)
        weights[cls_idx] = base_weight
    
    health_idx = None
    for name, idx in class_indices.items():
        if 'health' in name.lower() or 'healthy' in name.lower():
            health_idx = idx
            break
    
    if health_idx is not None:
        weights[health_idx] *= 1.5
        print(f"[!] Класс Healthy (idx={health_idx}) получает повышенный вес: {weights[health_idx]:.4f}")
    
    return weights


# =============================================================================
# ГЕНЕРАЦИЯ ОТЧЁТОВ
# =============================================================================
def generate_reports(model, valid_generator, history, class_names, output_dir):
    print("\n[*] Генерация предсказаний для отчётов...")
    
    valid_generator.reset()
    n_batches = len(valid_generator)
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    pbar = ProgressBar(total=n_batches, desc="Predictions")
    for i in range(n_batches):
        x, y = valid_generator[i]
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_pred_proba.extend(preds)
        pbar.update(1)
        if len(y_true) >= valid_generator.samples:
            break
    pbar.close()
    
    y_true = np.array(y_true[:valid_generator.samples])
    y_pred = np.array(y_pred[:valid_generator.samples])
    y_pred_proba = np.array(y_pred_proba[:valid_generator.samples])
    y_true_onehot = to_categorical(y_true, num_classes=len(class_names))
    
    print("[*] Сохранение отчётов и графиков...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] confusion_matrix.png")
    
    # 2. Class Metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'))
    
    # 3. Class Performance
    metrics_plot = report_df.loc[class_names, ['precision', 'recall', 'f1-score']]
    plt.figure(figsize=(16, 8))
    metrics_plot.plot(kind='bar', width=0.8)
    plt.title('Class Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] class_performance.png")
    
    # 4. Training Comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] training_comparison.png")
    
    # 5. Separate Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='orange')
    plt.title('Accuracy Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] accuracy_curves.png")
    
    # 6. Separate Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='red')
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='green')
    plt.title('Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] loss_curves.png")
    
    # 7. Training History
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    plt.figure(figsize=(14, 10))
    for col in history_df.columns:
        if col != 'epoch' and not col.startswith('val_'):
            plt.plot(history_df['epoch'], history_df[col], label=col, linewidth=2)
    plt.title('Training History - All Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] training_history.png")
    
    # 8. Learning Rate
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'], linewidth=2, color='purple')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.axhline(y=Config.INITIAL_LR, color='purple', linewidth=2, label='Initial LR')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print("  [+] learning_rate.png")
    
    # 9. Loss Distribution
    print("[*] Вычисление распределения потерь...")
    per_sample_loss = []
    valid_generator.reset()
    pbar = ProgressBar(total=n_batches, desc="Loss Distribution")
    for i in range(n_batches):
        x, y = valid_generator[i]
        preds = model.predict(x, verbose=0)
        for j in range(len(y)):
            true_idx = np.argmax(y[j])
            loss = -np.log(preds[j][true_idx] + 1e-7)
            per_sample_loss.append(loss)
        pbar.update(1)
        if len(per_sample_loss) >= valid_generator.samples:
            break
    pbar.close()
    
    per_sample_loss = per_sample_loss[:valid_generator.samples]
    plt.figure(figsize=(10, 6))
    plt.hist(per_sample_loss, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.title('Loss Distribution (Per Sample)', fontsize=14, fontweight='bold')
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(np.mean(per_sample_loss), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(per_sample_loss):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] loss_distribution.png")
    
    # 10. Prediction Confidence
    confidences = np.max(y_pred_proba, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence (Max Probability)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(confidences):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_confidence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] prediction_confidence.png")
    
    # 11. ROC Curves
    print("[*] Построение ROC-кривых...")
    plt.figure(figsize=(14, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                 label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] roc_curves.png")
    
    # 12. PR Curves
    print("[*] Построение PR-кривых...")
    plt.figure(figsize=(14, 12))
    for i, class_name in enumerate(class_names):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall_vals, precision_vals)
        plt.plot(recall_vals, precision_vals, color=colors[i], linewidth=2,
                 label=f'{class_name} (AP = {pr_auc:.3f})')
    
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(fontsize=9, loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [+] pr_curves.png")
    
    print(f"\n[+] Все отчёты сохранены в: {output_dir}")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    ensure_dirs()
    
    train_dir = os.path.join(Config.DATASET_ROOT, 'train')
    valid_dir = os.path.join(Config.DATASET_ROOT, 'valid')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Папка train не найдена: {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Папка valid не найдена: {valid_dir}")
    
    train_datagen = ImageDataGenerator(**Config.AUGMENTATION)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("[*] Загрузка train генератора...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    print("[*] Загрузка valid генератора...")
    valid_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    print(f"[*] Обнаружено классов: {num_classes}")
    print(f"[*] Классы: {class_names}")
    
    print("[*] Вычисление весов классов...")
    class_weights = compute_class_weights(train_generator)
    print(f"[*] Веса классов: { {k: round(v, 3) for k, v in class_weights.items()} }")
    
    latest_ckpt, start_epoch = get_latest_checkpoint()
    
    if latest_ckpt and os.path.exists(latest_ckpt):
        print(f"[!] Найден чекпоинт: {latest_ckpt}")
        print(f"[!] Продолжаем обучение с эпохи {start_epoch + 1}")
        model = load_model(latest_ckpt)
        initial_epoch = start_epoch
    else:
        print(f"[*] Чекпоинты не найдены. Создаём новую модель.")
        model = build_model(num_classes)
        initial_epoch = 0
    
    model.summary()
    
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, 'epoch_{epoch:03d}.h5')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=False,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=Config.MIN_LR,
        verbose=1
    )
    
    cleanup_callback = CheckpointCleanupCallback()
    training_logger = TrainingLogger()
    f1_callback = F1ValidationCallback(
        validation_generator=valid_generator,
        num_classes=num_classes,
        class_names=class_names
    )
    # FIX: Added memory cleanup callback to prevent OOM
    memory_cleanup = MemoryCleanupCallback()
    
    # НОВАЯ СТРАТЕГИЯ ОБУЧЕНИЯ: обучаем до достижения целевой точности по классам
    print(f"\n[*] Начало обучения (цель: recall >= {Config.TARGET_RECALL_PER_CLASS*100}% по всем классам)...")
    print(f"[*] Максимальное количество эпох: {Config.EPOCHS}")
    
    current_epoch = initial_epoch
    target_achieved = False
    
    try:
        while current_epoch < Config.EPOCHS and not target_achieved:
            # Обучаем одну эпоху
            history = model.fit(
                train_generator,
                initial_epoch=current_epoch,
                epochs=current_epoch+1,
                validation_data=valid_generator,
                class_weight=class_weights,
                callbacks=[checkpoint_callback, early_stopping, reduce_lr, 
                           cleanup_callback, training_logger, f1_callback,
                           memory_cleanup],  # FIX: Added memory cleanup
                verbose=1,
                workers=1,
                use_multiprocessing=False
            )
            current_epoch += 1
            
            # Вычисляем recall по классам на валидации
            valid_generator.reset()
            y_true = []
            y_pred = []
            n_batches = len(valid_generator)
            
            pbar = ProgressBar(total=n_batches, desc="Validation Recall Calc")
            for i in range(n_batches):
                x, y = valid_generator[i]
                preds = model.predict(x, verbose=0)
                y_true.extend(np.argmax(y, axis=1))
                y_pred.extend(np.argmax(preds, axis=1))
                pbar.update(1)
                if len(y_true) >= valid_generator.samples:
                    break
            pbar.close()
            
            y_true = np.array(y_true[:valid_generator.samples])
            y_pred = np.array(y_pred[:valid_generator.samples])
            
            # Вычисляем recall для каждого класса
            recalls = recall_score(y_true, y_pred, average=None, labels=range(num_classes))
            
            # Выводим текущие метрики
            print(f"\n[Epoch {current_epoch}] Recall по классам:")
            for i, (name, recall_val) in enumerate(zip(class_names, recalls)):
                print(f"  {name}: {recall_val:.4f} {'✓' if recall_val >= Config.TARGET_RECALL_PER_CLASS else '✗'}")
            
            # Проверяем условие остановки
            if np.all(recalls >= Config.TARGET_RECALL_PER_CLASS):
                target_achieved = True
                print(f"\n[+] ЦЕЛЕВАЯ ТОЧНОСТЬ ДОСТИЖЕНА! Recall >= {Config.TARGET_RECALL_PER_CLASS*100}% по всем классам на эпохе {current_epoch}")
            else:
                print(f"[-] Цель пока не достигнута. Продолжаем обучение...")
                
    except KeyboardInterrupt:
        print("\n[!] Обучение прервано пользователем")
    
    if not target_achieved and current_epoch >= Config.EPOCHS:
        print(f"\n[!] Достигнуто максимальное количество эпох ({Config.EPOCHS}) без выполнения условия")
    
    training_logger.save(os.path.join(Config.REPORTS_DIR, 'training_log.json'))
    generate_reports(model, valid_generator, history, class_names, Config.REPORTS_DIR)
    
    final_model_path = os.path.join(Config.MODELS_DIR, 'WheatDiseaseDetection.h5')
    model.save(final_model_path)
    print(f"\n[+] Модель сохранена: {final_model_path}")
    print(f"[+] Размер модели: {os.path.getsize(final_model_path) / (1024*1024):.2f} MB")
    
    print("\n[*] Финальная оценка модели...")
    valid_generator.reset()
    eval_results = model.evaluate(valid_generator, verbose=1)
    for name, value in zip(model.metrics_names, eval_results):
        print(f"    {name}: {value:.4f}")
    
    print("\n[*] Classification Report:")
    valid_generator.reset()
    y_true = []
    y_pred = []
    n_batches = len(valid_generator)
    pbar = ProgressBar(total=n_batches, desc="Final Evaluation")
    for i in range(n_batches):
        x, y = valid_generator[i]
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        pbar.update(1)
        if len(y_true) >= valid_generator.samples:
            break
    pbar.close()
    
    y_true = y_true[:valid_generator.samples]
    y_pred = y_pred[:valid_generator.samples]
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    health_idx = None
    for name, idx in train_generator.class_indices.items():
        if 'health' in name.lower() or 'healthy' in name.lower():
            health_idx = idx
            break
    
    if health_idx is not None:
        health_mask = np.array(y_true) == health_idx
        if np.any(health_mask):
            health_acc = np.mean(np.array(y_pred)[health_mask] == health_idx)
            print(f"\n[!] Точность класса Healthy: {health_acc:.4f} ({np.sum(health_mask)} образцов)")


if __name__ == '__main__':
    main()
