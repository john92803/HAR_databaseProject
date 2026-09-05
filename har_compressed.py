import time
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import struct
import zlib

start_time = time.time()

# TimescaleDB連接設定
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'har',
    'user': 'postgres',
    'password': '123'
}

# CSV檔案路徑
CSV_FILES = {
    'x': 'body_acc_x_train.csv',
    'y': 'body_acc_y_train.csv',
    'z': 'body_acc_z_train.csv'
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


class CompressionSchemes:
    """實作論文中的輕量級壓縮方案"""

    @staticmethod
    def delta_encode(values):
        """DELTA編碼：儲存相鄰值的差異"""
        if len(values) == 0:
            return [], None

        base = values[0]
        deltas = [0] + [values[i] - values[i-1] for i in range(1, len(values))]
        return deltas, base

    @staticmethod
    def delta_decode(deltas, base):
        """DELTA解碼"""
        if len(deltas) == 0:
            return []

        values = [base]
        for delta in deltas[1:]:
            values.append(values[-1] + delta)
        return values

    @staticmethod
    def rle_encode(values):
        """Run-Length Encoding：壓縮連續重複的值"""
        if len(values) == 0:
            return [], []

        encoded_values = []
        run_lengths = []
        current_value = values[0]
        current_length = 1

        for i in range(1, len(values)):
            if values[i] == current_value:
                current_length += 1
            else:
                encoded_values.append(current_value)
                run_lengths.append(current_length)
                current_value = values[i]
                current_length = 1

        # 添加最後一個run
        encoded_values.append(current_value)
        run_lengths.append(current_length)

        return encoded_values, run_lengths

    @staticmethod
    def rle_decode(encoded_values, run_lengths):
        """RLE解碼"""
        decoded = []
        for value, length in zip(encoded_values, run_lengths):
            decoded.extend([value] * length)
        return decoded

    @staticmethod
    def quantize_float(values, scale):
        """SCALE：將浮點數轉換為整數（論文中的SCALE方案）"""
        return [int(v * scale) for v in values], scale

    @staticmethod
    def dequantize_float(int_values, scale):
        """反量化"""
        return [v / scale for v in int_values]

    @staticmethod
    def compress_binary(data_bytes):
        """使用zlib進行二進制壓縮"""
        return zlib.compress(data_bytes, level=6)

    @staticmethod
    def decompress_binary(compressed_bytes):
        """zlib解壓縮"""
        return zlib.decompress(compressed_bytes)


class CompressedDataLoader:
    """壓縮資料載入器"""

    def __init__(self):
        self.compression = CompressionSchemes()

    def analyze_column_properties(self, df):
        """分析資料特性以選擇最佳壓縮方案"""
        properties = {}

        # 分析activity欄位
        properties['activity'] = {
            'type': 'categorical',
            'distinct_values': df['activity'].nunique(),
            'compression': 'dict'  # 使用字典編碼
        }

        # 分析subject欄位
        properties['subject'] = {
            'type': 'categorical',
            'distinct_values': df['subject'].nunique(),
            'compression': 'dict'
        }

        # 分析加速度資料欄位
        data_columns = [col for col in df.columns if col.startswith('Column')]
        sample_data = df[data_columns].values.flatten()[:10000]  # 取樣分析

        # 計算平均連續長度（用於RLE）
        run_lengths = []
        current_length = 1
        for i in range(1, len(sample_data)):
            if abs(sample_data[i] - sample_data[i-1]) < 0.001:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_length = 1
        avg_run_length = np.mean(run_lengths) if run_lengths else 1

        properties['acceleration'] = {
            'type': 'numeric',
            'avg_run_length': avg_run_length,
            'compression': 'delta+quantize+binary' if avg_run_length < 2 else 'rle+binary'
        }

        return properties

    def compress_window_data(self, window_data):
        """
        壓縮單個視窗的128個三軸加速度資料
        採用級聯壓縮：SCALE -> DELTA -> Binary Compression
        """
        x_data, y_data, z_data = window_data

        # 1. SCALE: 將浮點數轉為整數（保留3位小數精度）
        scale = 1000
        x_int, scale = self.compression.quantize_float(x_data, scale)
        y_int, _ = self.compression.quantize_float(y_data, scale)
        z_int, _ = self.compression.quantize_float(z_data, scale)

        # 2. DELTA: 編碼差異值（對於有序的時序資料很有效）
        x_deltas, x_base = self.compression.delta_encode(x_int)
        y_deltas, y_base = self.compression.delta_encode(y_int)
        z_deltas, z_base = self.compression.delta_encode(z_int)

        # 3. 打包為二進制格式
        # 格式: [scale(4字節)] [x_base(4)] [y_base(4)] [z_base(4)] [128*3個delta值]
        binary_data = struct.pack('i', scale)
        binary_data += struct.pack('i', x_base)
        binary_data += struct.pack('i', y_base)
        binary_data += struct.pack('i', z_base)

        # 使用短整數打包delta值（通常delta值較小）
        for x, y, z in zip(x_deltas, y_deltas, z_deltas):
            # 限制在-32768到32767範圍內
            x_clamped = max(-32768, min(32767, x))
            y_clamped = max(-32768, min(32767, y))
            z_clamped = max(-32768, min(32767, z))
            binary_data += struct.pack('hhh', x_clamped, y_clamped, z_clamped)

        # 4. Binary壓縮（使用zlib）
        compressed = self.compression.compress_binary(binary_data)

        return compressed

    def decompress_window_data(self, compressed_data):
        """解壓縮視窗資料"""
        # 1. Binary解壓縮
        binary_data = self.compression.decompress_binary(compressed_data)

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
        x_int = self.compression.delta_decode(x_deltas, x_base)
        y_int = self.compression.delta_decode(y_deltas, y_base)
        z_int = self.compression.delta_decode(z_deltas, z_base)

        # 5. 反量化
        x_data = self.compression.dequantize_float(x_int, scale)
        y_data = self.compression.dequantize_float(y_int, scale)
        z_data = self.compression.dequantize_float(z_int, scale)

        return x_data, y_data, z_data


def create_compressed_tables(conn):
    """創建壓縮資料表結構"""
    with conn.cursor() as cur:
        # 嘗試啟用TimescaleDB擴展
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            conn.commit()
            use_timescale = True
            print("timescaledb extension enabled.")
        except Exception as e:
            print(f"無法啟用TimescaleDB擴展: {e}")
            print("將使用普通PostgreSQL表格")
            use_timescale = False
            conn.rollback()

        # 刪除舊表（如果需要重新創建）
        cur.execute("DROP TABLE IF EXISTS activity_data CASCADE;")
        conn.commit()

        # 創建壓縮資料表（按視窗儲存）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS activity_data_compressed (
                window_id INTEGER PRIMARY KEY,
                time TIMESTAMPTZ NOT NULL,
                subject_id SMALLINT NOT NULL,
                activity_id SMALLINT NOT NULL,
                activity_name VARCHAR(50) NOT NULL,
                compressed_data BYTEA NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL GENERATED ALWAYS AS 
                    (CAST(compressed_size AS REAL) / NULLIF(original_size, 0) * 100.0) STORED
            );
        """)

        # 創建索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_compressed_subject_activity 
            ON activity_data_compressed (subject_id, activity_id, time DESC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_compressed_time 
            ON activity_data_compressed (time DESC);
        """)

        # 創建統計視圖
        cur.execute("""
            CREATE OR REPLACE VIEW compression_statistics AS
            SELECT 
                activity_name,
                COUNT(*) as window_count,
                ROUND(AVG(original_size)::numeric, 2) as avg_original_size,
                ROUND(AVG(compressed_size)::numeric, 2) as avg_compressed_size,
                ROUND(AVG(compression_ratio)::numeric, 2) as avg_compression_ratio,
                SUM(original_size) as total_original_size,
                SUM(compressed_size) as total_compressed_size,
                ROUND((SUM(compressed_size)::numeric / NULLIF(SUM(original_size), 0) * 100.0)::numeric, 2) as actual_ratio
            FROM activity_data_compressed
            GROUP BY activity_name
            ORDER BY activity_name;
        """)

        conn.commit()
        print("compressed tables created successfully.")


def load_and_compress_data():
    """載入CSV並壓縮儲存"""
    print("loading and compressing data...")

    # 讀取三個CSV檔案
    df_x = pd.read_csv(CSV_FILES['x'],
                       skipinitialspace=True,
                       skip_blank_lines=True)
    df_y = pd.read_csv(CSV_FILES['y'],
                       skipinitialspace=True,
                       skip_blank_lines=True)
    df_z = pd.read_csv(CSV_FILES['z'],
                       skipinitialspace=True,
                       skip_blank_lines=True)

    # 移除空白行
    df_x = df_x.dropna(how='all')
    df_y = df_y.dropna(how='all')
    df_z = df_z.dropna(how='all')

    # 確保行數一致
    min_rows = min(len(df_x), len(df_y), len(df_z))
    df_x = df_x.iloc[:min_rows].reset_index(drop=True)
    df_y = df_y.iloc[:min_rows].reset_index(drop=True)
    df_z = df_z.iloc[:min_rows].reset_index(drop=True)



    # 分析資料特性
    loader = CompressedDataLoader()
    properties = loader.analyze_column_properties(df_x)
    # print(f"\n資料特性分析:")
    for key, value in properties.items():
        print(f"  {key}: {value}")

    # 準備壓縮資料
    data_columns = [col for col in df_x.columns if col.startswith('Column')]
    records = []
    base_time = datetime(2023, 1, 1)

    total_original_size = 0
    total_compressed_size = 0

    for idx in range(min_rows):
        if idx % 100 == 0:
            print(f"進度: {idx}/{min_rows}")

        # 提取三軸資料
        x_data = df_x.iloc[idx][data_columns].values.astype(float)
        y_data = df_y.iloc[idx][data_columns].values.astype(float)
        z_data = df_z.iloc[idx][data_columns].values.astype(float)

        # 壓縮視窗資料
        compressed = loader.compress_window_data((x_data, y_data, z_data))

        # 計算壓縮比
        original_size = 128 * 3 * 4  # 128樣本 * 3軸 * 4字節(float)
        compressed_size = len(compressed)

        total_original_size += original_size
        total_compressed_size += compressed_size

        # 準備插入資料
        activity_id = int(df_x.iloc[idx]['activity'])
        subject_id = int(df_x.iloc[idx]['subject'])
        activity_name = ACTIVITY_LABELS.get(activity_id, 'UNKNOWN')
        window_time = base_time + timedelta(seconds=idx * 2.56)

        records.append((
            idx,  # window_id
            window_time,
            subject_id,
            activity_id,
            activity_name,
            psycopg2.Binary(compressed),
            original_size,
            compressed_size
        ))

    overall_ratio = total_compressed_size / total_original_size
    print(f"\ntotal compression ratio: {overall_ratio:.2%}")
    print(f"original size: {total_original_size / 1024 / 1024:.2f} MB")
    print(f"compressed size: {total_compressed_size / 1024 / 1024:.2f} MB")
    print(f"size reduction: {(1 - overall_ratio) * 100:.2f}%")

    return records


def insert_compressed_data(conn, records, batch_size=10000):
    """批次插入壓縮資料"""

    insert_query = """
        INSERT INTO activity_data_compressed 
        (window_id, time, subject_id, activity_id, activity_name, 
         compressed_data, original_size, compressed_size)
        VALUES %s
        ON CONFLICT (window_id) DO NOTHING
    """

    with conn.cursor() as cur:
        total_batches = (len(records) + batch_size - 1) // batch_size

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            execute_values(cur, insert_query, batch)
            conn.commit()

            current_batch = i // batch_size + 1
            print(f"已插入批次 {current_batch}/{total_batches}")

    print("compressed data insertion complete.")


# def verify_compression(conn):
#     """驗證壓縮效果"""
#     print("\n驗證壓縮效果...")

#     with conn.cursor() as cur:
#         # 隨機選取一個視窗進行解壓縮測試
#         cur.execute("""
#             SELECT window_id, compressed_data, activity_name
#             FROM activity_data_compressed
#             ORDER BY RANDOM()
#             LIMIT 1
#         """)

#         row = cur.fetchone()
#         if row:
#             window_id, compressed_data, activity_name = row

#             # 解壓縮
#             loader = CompressedDataLoader()
#             x_data, y_data, z_data = loader.decompress_window_data(
#                 bytes(compressed_data))

#             print(f"\n解壓縮測試 (Window {window_id}, Activity: {activity_name}):")
#             print(f"  X軸前5個值: {x_data[:5]}")
#             print(f"  Y軸前5個值: {y_data[:5]}")
#             print(f"  Z軸前5個值: {z_data[:5]}")
#             print("  ✓ 解壓縮成功")


def show_statistics(conn):
    """顯示壓縮統計資訊"""


    with conn.cursor() as cur:
        # 總體統計
        cur.execute("""
            SELECT 
                COUNT(*) as total_windows,
                SUM(original_size) / 1024.0 / 1024.0 as total_original_mb,
                SUM(compressed_size) / 1024.0 / 1024.0 as total_compressed_mb,
                (SUM(compressed_size)::float / NULLIF(SUM(original_size), 0) * 100.0) as overall_ratio
            FROM activity_data_compressed
        """)
        row = cur.fetchone()
        print(f"\ntotal compression statistics:")
        print(f"  window_count: {row[0]:,}")
        print(f"  original: {row[1]:.2f} MB")
        print(f"  compressed: {row[2]:.2f} MB")
        print(f"  compression_ratio: {row[3]:.2f}%")
        print(f"  saving: {(100 - row[3]):.2f}%")

        # 各活動的壓縮統計
        cur.execute("SELECT * FROM compression_statistics")
        print("\neach activity compression statistics:")
        print(
            f"{'activity':<25} {'window_count':<10} {'original(KB)':<12} {'compressed(KB)':<12} {'compression_ratio':<12} {'saving':<10}")
        print("-" * 85)
        for row in cur.fetchall():
            activity_name = row[0]
            window_count = row[1]
            total_orig_kb = row[5] / 1024.0
            total_comp_kb = row[6] / 1024.0
            ratio = row[7]
            saving = 100 - ratio
            print(f"{activity_name:<25} {window_count:<10} {total_orig_kb:<12.2f} {total_comp_kb:<12.2f} {ratio:<12.2f}% {saving:<10.2f}%")


def main():
    """主程式"""
    try:
        # 連接資料庫
        conn = psycopg2.connect(**DB_CONFIG)

        # 創建表格
        create_compressed_tables(conn)

        # 載入並壓縮資料
        records = load_and_compress_data()

        # 插入資料
        insert_compressed_data(conn, records)

        # # 驗證壓縮
        # verify_compression(conn)

        # 顯示統計資訊
        show_statistics(conn)

        conn.close()
        end_time = time.time()
        execution_time = end_time - start_time
        print("time：", int(execution_time), "sec")

    except Exception as e:
        print(f"error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
