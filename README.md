# HAR Database Project

以 TimescaleDB 儲存三軸加速度時序資料，比較「原始儲存」與「壓縮儲存」兩種方案在儲存空間、讀寫時間與模型準確率上的差異。資料集為 UCI Human Activity Recognition Using Smartphones，任務為六類人體活動辨識。

## 功能

- **原始資料匯入** — 將每個取樣點展開成一列存入資料庫，依 50 Hz 計算時間戳記，建立 hypertable 與索引後批次寫入
- **壓縮資料匯入** — 以視窗為單位，將 128×3 筆資料經 SCALE → DELTA → zlib 三階段級聯壓縮，存成單一 `BYTEA` 欄位，並記錄原始／壓縮大小與壓縮率
- **模型訓練（原始）** — 從原始表查詢資料，訓練 1D CNN 進行六類活動分類
- **模型訓練（壓縮）** — 從壓縮表讀出後解壓還原，以相同架構訓練並比較準確率
- **成效統計** — 以 SQL 視圖依活動類別彙整壓縮率，並輸出訓練曲線與混淆矩陣

## 資料

UCI HAR Dataset，訓練集整理為三個 CSV，每列為一個 2.56 秒視窗、含 128 個取樣點（50 Hz），附 `subject` 與 `activity` 欄位。六類活動為 `WALKING`、`WALKING_UPSTAIRS`、`WALKING_DOWNSTAIRS`、`SITTING`、`STANDING`、`LAYING`。

下載：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## 檔案結構

```
readData.py           # 原始資料匯入 TimescaleDB
har_compressed.py     # 壓縮方案實作與壓縮資料匯入
train.py              # 從原始表讀取資料訓練
decomp_train.py       # 從壓縮表解壓後訓練
body_acc_x_train.csv
body_acc_y_train.csv
body_acc_z_train.csv
result/               # 實驗結果圖表
```
