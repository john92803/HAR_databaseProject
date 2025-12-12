import time
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import numpy as np

start_time = time.time()

# TimescaleDB連接設定
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'har',
    'user': '',
    'password': ''
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


def create_tables(conn):
    """創建TimescaleDB表格"""
    with conn.cursor() as cur:
        # 嘗試啟用TimescaleDB擴展
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            conn.commit()
            use_timescale = True
            print("TimescaleDB擴展已啟用")
        except Exception as e:
            print(f"無法啟用TimescaleDB擴展: {e}")
            print("將使用普通PostgreSQL表格")
            use_timescale = False
            conn.rollback()
        
        # 刪除舊表（如果需要重新創建）
        cur.execute("DROP TABLE IF EXISTS activity_data CASCADE;")
        conn.commit()
        
        # 創建主表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS activity_data (
                time TIMESTAMPTZ NOT NULL,
                subject_id INTEGER NOT NULL,
                activity_id INTEGER NOT NULL,
                activity_name VARCHAR(50) NOT NULL,
                window_id INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                acc_x DOUBLE PRECISION,
                acc_y DOUBLE PRECISION,
                acc_z DOUBLE PRECISION
            );
        """)
        conn.commit()
        print("資料表創建成功")
        
        # 如果可用，創建超表（hypertable）
        if use_timescale:
            try:
                # 檢查表是否已經是hypertable
                cur.execute("""
                    SELECT * FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = 'activity_data';
                """)
                if cur.fetchone() is None:
                    cur.execute("""
                        SELECT create_hypertable('activity_data', 'time', 
                                                 if_not_exists => TRUE);
                    """)
                    conn.commit()
                    print("Hypertable創建成功")
                else:
                    print("表格已經是hypertable")
            except Exception as e:
                print(f"創建hypertable時出錯: {e}")
                conn.rollback()
        
        # 創建索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject_activity 
            ON activity_data (subject_id, activity_id, time DESC);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_time 
            ON activity_data (time DESC);
        """)
        
        conn.commit()
        print("indexes created successfully.")


def load_csv_files():
    """讀取三個CSV檔案"""

    df_x = pd.read_csv(CSV_FILES['x'], 
                       skipinitialspace=True,
                       skip_blank_lines=True,
                       encoding='utf-8')
    df_y = pd.read_csv(CSV_FILES['y'], 
                       skipinitialspace=True,
                       skip_blank_lines=True,
                       encoding='utf-8')
    df_z = pd.read_csv(CSV_FILES['z'], 
                       skipinitialspace=True,
                       skip_blank_lines=True,
                       encoding='utf-8')
    

    # 驗證三個檔案行數相同
    if len(df_x) != len(df_y) or len(df_y) != len(df_z):
        print("\nerror: csv files have different number of rows.")
        print(f"difference: X={len(df_x)}, Y={len(df_y)}, Z={len(df_z)}")
        
        # 找出最小行數
        min_rows = min(len(df_x), len(df_y), len(df_z))
        
        df_x = df_x.iloc[:min_rows].reset_index(drop=True)
        df_y = df_y.iloc[:min_rows].reset_index(drop=True)
        df_z = df_z.iloc[:min_rows].reset_index(drop=True)
    
    return df_x, df_y, df_z


def prepare_data(df_x, df_y, df_z, base_time=None):
    """準備要插入資料庫的資料"""
    if base_time is None:
        base_time = datetime(2025, 1, 1)
    
    # 取得資料欄位（Column1~128）
    data_columns = [col for col in df_x.columns if col.startswith('Column')]
    num_samples = len(data_columns)
    
    # 採樣頻率50Hz，每個樣本間隔0.02秒
    sample_interval = timedelta(seconds=0.02)
    
    records = []
    window_id = 0
    
    print("processing data...")
    for idx in range(len(df_x)):
        if idx % 100 == 0:
            print(f"processing row: {idx}/{len(df_x)}")
        
        activity_id = int(df_x.iloc[idx]['activity'])
        subject_id = int(df_x.iloc[idx]['subject'])
        activity_name = ACTIVITY_LABELS.get(activity_id, 'UNKNOWN')
        
        # 每個視窗的起始時間
        window_start_time = base_time + timedelta(seconds=window_id * 2.56)
        
        # 處理128個樣本點
        for sample_idx, col in enumerate(data_columns):
            timestamp = window_start_time + sample_interval * sample_idx
            
            acc_x = float(df_x.iloc[idx][col])
            acc_y = float(df_y.iloc[idx][col])
            acc_z = float(df_z.iloc[idx][col])
            
            records.append((
                timestamp,
                subject_id,
                activity_id,
                activity_name,
                window_id,
                sample_idx,
                acc_x,
                acc_y,
                acc_z
            ))
        
        window_id += 1
    
    print(f"total records: {len(records)} ")
    return records


def insert_data(conn, records, batch_size=10000):
    """批次插入資料到TimescaleDB"""

    insert_query = """
        INSERT INTO activity_data 
        (time, subject_id, activity_id, activity_name, window_id, 
         sample_index, acc_x, acc_y, acc_z)
        VALUES %s
    """
    
    with conn.cursor() as cur:
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            execute_values(cur, insert_query, batch)
            conn.commit()
            
            current_batch = i // batch_size + 1
            print(f"inserted {current_batch}/{total_batches}")
    
    print("data insertion completed.")


# def create_summary_views(conn):
#     """創建一些有用的視圖"""
#     with conn.cursor() as cur:
#         # 創建每個視窗的統計視圖
#         cur.execute("""
#             CREATE MATERIALIZED VIEW IF NOT EXISTS window_statistics AS
#             SELECT 
#                 window_id,
#                 subject_id,
#                 activity_name,
#                 MIN(time) as window_start,
#                 MAX(time) as window_end,
#                 AVG(acc_x) as avg_acc_x,
#                 AVG(acc_y) as avg_acc_y,
#                 AVG(acc_z) as avg_acc_z,
#                 STDDEV(acc_x) as std_acc_x,
#                 STDDEV(acc_y) as std_acc_y,
#                 STDDEV(acc_z) as std_acc_z,
#                 SQRT(AVG(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z)) as magnitude_mean
#             FROM activity_data
#             GROUP BY window_id, subject_id, activity_name;
#         """)
        
#         conn.commit()
#         print("統計視圖創建成功！")


def main():
    """主程式"""
    try:
        # 讀取CSV檔案
        df_x, df_y, df_z = load_csv_files()
        
        # 連接資料庫
        print("正在連接TimescaleDB...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("資料庫連接成功！")
        
        # 創建表格
        create_tables(conn)
        
        # 準備資料
        records = prepare_data(df_x, df_y, df_z)
        
        # 插入資料
        insert_data(conn, records)
        
        # # 創建統計視圖
        # create_summary_views(conn)
        
        # 顯示統計資訊
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM activity_data;")
            total_records = cur.fetchone()[0]
            print(f"total records: {total_records:,}")
            
            cur.execute("""
                SELECT activity_name, COUNT(*) as count 
                FROM activity_data 
                GROUP BY activity_name 
                ORDER BY count DESC;
            """)
            print("\each activity statistics:")
            for row in cur.fetchall():
                print(f"  {row[0]}: {row[1]:,}")
            
            cur.execute("""
                SELECT subject_id, COUNT(DISTINCT window_id) as windows 
                FROM activity_data 
                GROUP BY subject_id 
                ORDER BY subject_id;
            """)
            # print("\n各受試者的視窗數:")
            # for row in cur.fetchall():
            #     print(f"  受試者 {row[0]}: {row[1]} 個視窗")
        
        conn.close()
        end_time = time.time()
        execution_time = end_time - start_time
        print("done")
        print("time：", int(execution_time), "秒")
        
    except Exception as e:
        print(f"error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
