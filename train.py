import time
import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

start_time = time.time()

# 資料庫連接設定
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'har',
    'user': '',
    'password': ''
}

# 活動標籤映射
ACTIVITY_LABELS = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}


def load_data_from_database():
    """從資料庫讀取資料"""
    conn = psycopg2.connect(**DB_CONFIG)

    # 查詢資料，按window_id和sample_index排序
    query = """
        SELECT 
            activity_id,
            activity_name,
            window_id,
            sample_index,
            acc_x,
            acc_y,
            acc_z
        FROM activity_data
        ORDER BY window_id, sample_index;
    """

    print("processing...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"complete: {len(df)} ")
    # print(f"視窗數量: {df['window_id'].nunique()}")
    # print(f"活動類型: {df['activity_name'].unique()}")

    return df


def prepare_training_data(df):
    """準備訓練資料"""
    print("preparing training data...")

    # 按window_id分組，每個視窗應該有128個樣本
    windows = []
    labels = []
    activity_names = []

    for window_id, group in df.groupby('window_id'):
        # 確保每個視窗有完整的128個樣本
        if len(group) == 128:
            # 提取三軸加速度數據
            acc_data = group[['acc_x', 'acc_y', 'acc_z']].values
            windows.append(acc_data)

            # 標籤（使用第一個樣本的標籤，因為同一視窗的標籤都相同）
            labels.append(group['activity_id'].iloc[0])
            activity_names.append(group['activity_name'].iloc[0])

    # 轉換為numpy陣列
    X = np.array(windows)  # shape: (num_windows, 128, 3)
    y = np.array(labels) - 1  # 將1-6轉換為0-5（模型需要從0開始的標籤）

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"activity distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        activity_name = ACTIVITY_LABELS[label + 1]
        print(f"window:  {activity_name}: {count} ")

    return X, y


def create_model(input_shape, num_classes):
    """創建CNN模型"""
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


def plot_training_history(history):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 準確率
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # 損失
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("training_history saved: training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, activity_labels):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_labels,
                yticklabels=activity_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("confusion_matrix saved: confusion_matrix.png")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """評估模型"""
    print("\n evaluating model...")

    # 預測
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 評估指標
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n test accuracy: {test_accuracy:.4f}")
    print(f"test loss: {test_loss:.4f}")

    # 詳細分類報告
    activity_labels = [ACTIVITY_LABELS[i+1] for i in range(6)]
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, target_names=activity_labels))

    # 繪製混淆矩陣
    plot_confusion_matrix(y_test, y_pred, activity_labels)

    return y_pred


def main():
    """主程式"""
    try:
        # 1. 從資料庫讀取資料
        df = load_data_from_database()

        # 2. 準備訓練資料
        X, y = prepare_training_data(df)

        # 3. 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\ntraining set size: {X_train.shape[0]}")
        print(f"test set size: {X_test.shape[0]}")

        # 4. 創建模型
        print("\ncreating model...")
        num_classes = len(ACTIVITY_LABELS)
        model = create_model(input_shape=(128, 3), num_classes=num_classes)
        model.summary()

        # 5. 設定回調函數
        earlystop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        )

        checkpoint = ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # 6. 訓練模型
        print("\nstarting training...")
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[earlystop, checkpoint],
            verbose=1
        )

        # 7. 繪製訓練歷史
        plot_training_history(history)

        # 8. 評估模型
        y_pred = evaluate_model(model, X_test, y_test)

        # 9. 儲存最終模型
        model.save('final_model.keras')
        print("\nmodel saved: final_model.keras")

        # 10. 顯示一些預測範例
        print("\npredictions:")
        for i in range(min(10, len(X_test))):
            true_label = ACTIVITY_LABELS[y_test[i] + 1]
            pred_label = ACTIVITY_LABELS[y_pred[i] + 1]
            correct = "✓" if y_test[i] == y_pred[i] else "✗"
            print(f"{correct} true positive: {true_label:20s} | predicted: {pred_label:20s}")

        end_time = time.time()
        execution_time = end_time - start_time
        print("\ntrain complete！")
        print("time：", int(execution_time), "seconds")

    except Exception as e:
        print(f"error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
