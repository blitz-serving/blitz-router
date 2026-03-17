import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 若在服务器/无GUI环境，确保能保存图片
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv('/data/huggingface/datasets/2min.csv')

# 解析时间戳
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# 按 5 秒聚合（每5秒一个桶）
s = (
    df.set_index('TIMESTAMP')
      .resample('s')      # ← 改为每5秒
      .size()
      .rename('requests')
)

# 截取前 20 分钟（从最早时间开始的 20 分钟，按5秒补齐）
if not s.empty:
    start = s.index.min().floor('s')
    # 20分钟 / 5秒 = 240 个采样点
    full_idx = pd.date_range(start=start, periods=240, freq='s')
    s = s.reindex(full_idx, fill_value=0)

# 绘图（不显示数据点）
plt.figure(figsize=(12, 5))
plt.plot(s.index, s.values)  # 无 marker
plt.title('requests per 5s (first 20 minutes)')
plt.xlabel('time')
plt.ylabel('requests / 5s')
plt.grid(True)
plt.tight_layout()
plt.savefig('timeline_first_20min_5s.png', dpi=300)
print('saved: timeline_first_20min_5s.png')
