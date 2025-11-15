import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import struct
import zlib
from tqdm import tqdm

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


class CompressedDataReader:
    """壓縮資料讀取器"""
    
    @staticmethod
    def decompress_window_data(compressed_data):
        """解壓縮視窗資料"""
        try:
            # 1. Binary解壓縮
            binary_data = zlib.decompress(compressed_data)
            
            # 2. 解包header
            offset = 0
            scale = struct.unpack_from('i', binary_data, offset)[0]
            offset += 4
            x_base = struct.unpack_from('i', binary_data, offset)[0]
            offset += 4
            y_base = struct.unpack_from('i', binary_data, offset)[0]
            offset += 4
            z_base = struct.unpack_from('i', binary_data, offset)[0]
            offset += 4
            
            # 3. 解包delta值
            x_deltas, y_deltas, z_deltas = [], [], []
            for _ in range(128):
                x, y, z = struct.unpack_from('hhh', binary_data, offset)
                x_deltas.append(x)
                y_deltas.append(y)
                z_deltas.append(z)
                offset += 6
            
            # 4. DELTA解碼
            x_int = [x_base]
            for delta in x_deltas[1:]:
                x_int.append(x_int[-1] + delta)
            
            y_int = [y_base]
            for delta in y_deltas[1:]:
                y_int.append(y_int[-1] + delta)
            
            z_int = [z_base]
            for delta in z_deltas[1:]:
                z_int.append(z_int[-1] + delta)
            
            # 5. 反量化
            x_data = [v / scale for v in x_int]
            y_data = [v / scale for v in y_int]
            z_data = [v / scale for v in z_int]
            
            return np.array(x_data), np.array(y_data), np.array(z_data)
        
        except Exception as e:
            print(f"解壓縮錯誤: {e}")
            return None, None, None


def load_compressed_data_from_database(batch_size=1000):
    """
    從壓縮資料庫批次讀取資料
    使用批次處理避免記憶體溢出
    """
    print("正在連接資料庫...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # 先獲取總數
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM activity_data_compressed;")
        total_windows = cur.fetchone()[0]
    
    print(f"資料庫中共有 {total_windows} 個視窗")
    print("正在解壓縮並載入資料...")
    
    reader = CompressedDataReader()
    
    X_list = []
    y_list = []
    
    # 批次讀取
    with conn.cursor(name='compressed_cursor') as cur:
        cur.itersize = batch_size
        cur.execute("""
            SELECT 
                window_id,
                activity_id,
                compressed_data
            FROM activity_data_compressed
            ORDER BY window_id;
        """)
        
        batch_count = 0
        for row in tqdm(cur, total=total_windows, desc="解壓縮進度"):
            window_id, activity_id, compressed_data = row
            
            # 解壓縮
            x_data, y_data, z_data = reader.decompress_window_data(bytes(compressed_data))
            
            if x_data is not None:
                # 組合三軸資料 (128, 3)
                window_data = np.column_stack([x_data, y_data, z_data])
                X_list.append(window_data)
                y_list.append(activity_id - 1)  # 轉換為0-5
            
            batch_count += 1
            
            # 每1000個視窗顯示一次進度
            if batch_count % 1000 == 0:
                print(f"已處理 {batch_count}/{total_windows} 個視窗")
    
    conn.close()
    
    # 轉換為numpy陣列
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n載入完成!")
    print(f"X shape: {X.shape}")  # (num_windows, 128, 3)
    print(f"y shape: {y.shape}")  # (num_windows,)
    
    # 顯示活動分布
    print("\n活動分布:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        activity_name = ACTIVITY_LABELS[label + 1]
        print(f"  {activity_name}: {count} 個視窗 ({count/len(y)*100:.1f}%)")
    
    return X, y


def load_compressed_data_memory_efficient():
    """
    記憶體高效版本：使用生成器逐個讀取
    適合記憶體有限的情況
    """
    print("正在連接資料庫...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM activity_data_compressed;")
        total_windows = cur.fetchone()[0]
    
    print(f"將使用生成器模式載入 {total_windows} 個視窗")
    
    reader = CompressedDataReader()
    
    def data_generator():
        """資料生成器"""
        with conn.cursor(name='gen_cursor') as cur:
            cur.itersize = 100
            cur.execute("""
                SELECT 
                    window_id,
                    activity_id,
                    compressed_data
                FROM activity_data_compressed
                ORDER BY window_id;
            """)
            
            for row in cur:
                window_id, activity_id, compressed_data = row
                x_data, y_data, z_data = reader.decompress_window_data(bytes(compressed_data))
                
                if x_data is not None:
                    window_data = np.column_stack([x_data, y_data, z_data])
                    yield window_data, activity_id - 1
    
    return data_generator, total_windows, conn


def create_model(input_shape, num_classes):
    """創建CNN模型"""
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
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
    plt.savefig('training_history_compressed.png', dpi=300, bbox_inches='tight')
    print("訓練歷史圖已儲存: training_history_compressed.png")
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
    plt.savefig('confusion_matrix_compressed.png', dpi=300, bbox_inches='tight')
    print("混淆矩陣已儲存: confusion_matrix_compressed.png")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """評估模型"""
    print("\n正在評估模型...")
    
    # 預測
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 評估指標
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n測試集準確率: {test_accuracy:.4f}")
    print(f"測試集損失: {test_loss:.4f}")
    
    # 詳細分類報告
    activity_labels = [ACTIVITY_LABELS[i+1] for i in range(6)]
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=activity_labels))
    
    # 繪製混淆矩陣
    plot_confusion_matrix(y_test, y_pred, activity_labels)
    
    return y_pred


def main_standard():
    """標準訓練流程：一次性載入所有資料"""
    try:
        print("=" * 60)
        print("從壓縮資料庫訓練HAR模型")
        print("=" * 60)
        
        # 1. 從資料庫載入並解壓縮資料
        X, y = load_compressed_data_from_database()
        
        # 2. 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n訓練集大小: {X_train.shape[0]}")
        print(f"測試集大小: {X_test.shape[0]}")
        
        # 3. 創建模型
        print("\n創建模型...")
        num_classes = len(ACTIVITY_LABELS)
        model = create_model(input_shape=(128, 3), num_classes=num_classes)
        model.summary()
        
        # 4. 設定回調函數
        earlystop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_model_compressed.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # 5. 訓練模型
        print("\n開始訓練模型...")
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[earlystop, checkpoint],
            verbose=1
        )
        
        # 6. 繪製訓練歷史
        plot_training_history(history)
        
        # 7. 評估模型
        y_pred = evaluate_model(model, X_test, y_test)
        
        # 8. 儲存最終模型
        model.save('final_model_compressed.keras')
        print("\n模型已儲存: final_model_compressed.keras")
        
        # 9. 顯示一些預測範例
        print("\n預測範例:")
        for i in range(min(10, len(X_test))):
            true_label = ACTIVITY_LABELS[y_test[i] + 1]
            pred_label = ACTIVITY_LABELS[y_pred[i] + 1]
            correct = "✓" if y_test[i] == y_pred[i] else "✗"
            print(f"{correct} 真實: {true_label:20s} | 預測: {pred_label:20s}")
        
        print("\n訓練完成！")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


def main_memory_efficient():
    """
    記憶體高效版本：使用生成器逐批載入
    適合記憶體有限或資料量很大的情況
    """
    try:
        print("=" * 60)
        print("從壓縮資料庫訓練HAR模型（記憶體高效版）")
        print("=" * 60)
        
        # 1. 先載入所有資料做train/test split
        # （如果記憶體真的很小，需要預先在資料庫中標記train/test）
        print("正在載入資料用於分割...")
        data_gen, total_windows, conn = load_compressed_data_memory_efficient()
        
        # 收集所有資料
        X_list = []
        y_list = []
        
        print("第一次掃描：收集所有資料...")
        for window_data, label in tqdm(data_gen(), total=total_windows):
            X_list.append(window_data)
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n訓練集大小: {X_train.shape[0]}")
        print(f"測試集大小: {X_test.shape[0]}")
        
        # 後續步驟與標準版本相同
        num_classes = len(ACTIVITY_LABELS)
        model = create_model(input_shape=(128, 3), num_classes=num_classes)
        
        earlystop = EarlyStopping(monitor='val_loss', patience=10, 
                                  restore_best_weights=True, min_delta=0.001, verbose=1)
        checkpoint = ModelCheckpoint('best_model_compressed.keras', monitor='val_accuracy',
                                    save_best_only=True, mode='max', verbose=1)
        
        print("\n開始訓練...")
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[earlystop, checkpoint],
            verbose=1
        )
        
        plot_training_history(history)
        evaluate_model(model, X_test, y_test)
        model.save('final_model_compressed.keras')
        
        conn.close()
        print("\n訓練完成！")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 選擇訓練方式
    print("請選擇訓練方式:")
    print("1. 標準模式（一次性載入所有資料）")
    print("2. 記憶體高效模式（使用生成器）")
    
    choice = input("請輸入選項 (1 或 2，直接Enter使用標準模式): ").strip()
    
    if choice == "2":
        main_memory_efficient()
    else:
        main_standard()
