# import streamlit as st # type: ignore
# import pandas as pd # type: ignore
# import numpy as np # type: ignore
# import matplotlib.pyplot as plt # type: ignore

# # ============ 页面配置 ============
# st.set_page_config(page_title="Alibaba CTR Dashboard", layout="wide")
# st.markdown("## 🛍️ Alibaba CTR Prediction — Business Overview")
# st.caption("Based on full 1M-row sample · Overall CTR baseline = 5.16%  \nUCLA MDSH Capstone 2026")

# with st.expander("ℹ️ About this dashboard"):
#     st.markdown("""
#     This dashboard explores user behavior patterns on Alibaba's Taobao advertising dataset
#     to inform CTR prediction modeling. Every chart is designed to answer: 
#     *"What does this tell us for feature engineering?"*
    
#     **How to use**: Apply filters in the left sidebar to slice by gender, age, or shopping depth.  
#     **Data**: 1,023,313 impressions · 408,578 unique users · 8-day window (May 6–13, 2017).
#     """)

# # ============ 常量:label 映射(来自 2b)============
# DIMENSIONS = {
#     'final_gender_code': {1: 'Male', 2: 'Female'},
#     'age_level':         {1:'<18', 2:'18-24', 3:'25-29', 4:'30-34',
#                           5:'35-39', 6:'40-49', 7:'50+'},
#     'pvalue_level':      {1: 'Low', 2: 'Mid', 3: 'High'},
#     'shopping_level':    {1: 'Shallow', 2: 'Moderate', 3: 'Deep'},
#     'occupation':        {0: 'Non-Student', 1: 'Student'},
#     'new_user_class_level': {1:'Tier-1', 2:'Tier-2', 3:'Tier-3', 4:'Rural/Other'}
# }
# MIN_IMPRESSIONS = 500  # 来自 2b,小样本过滤阈值
# BASELINE_CTR = 5.16    # 来自 notebook 的整体 CTR
# NICE_NAMES = {
#     'age_level': 'Age Group',
#     'final_gender_code': 'Gender',
#     'pvalue_level': 'Consumption Level',
#     'shopping_level': 'Shopping Depth',
#     'new_user_class_level': 'City Tier',
#     'occupation': 'Occupation'
# } #让读者更好理解
# # ============ 数据加载 ============
# @st.cache_data
# def load_data():
#     df = pd.read_parquet("data/ctr_full.parquet")
    

# # 时间处理:从 time_stamp (UNIX 秒) 构造
#     df['event_time'] = pd.to_datetime(df['time_stamp'], unit='s', utc=True)
#     df['event_time_beijing'] = df['event_time'].dt.tz_convert('Asia/Shanghai')
#     df['hour'] = df['event_time_beijing'].dt.hour
#     df['day_of_week'] = df['event_time_beijing'].dt.dayofweek
#     df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
#     # Price 处理(来自 2c)
#     p99 = df['price'].quantile(0.99)
#     df['price_capped'] = df['price'].clip(upper=p99)
#     df['price_bucket'] = pd.cut(
#         df['price_capped'],
#         bins=[0, 100, 300, 600, 1320, p99 + 1],
#         labels=['0-100', '100-300', '300-600', '600-1320', '1320-5900']
#     )
    
#     # Profile matched flag(来自 2b 的巧思)
#     df['is_profile_matched'] = df['age_level'].notna().astype(int)
    
#     return df

# df = load_data()

# # ============ Sidebar Filters ============
# st.sidebar.header("🔍 Filters")
# st.sidebar.caption("Filters apply to all charts below.")

# # Gender filter
# gender_opts = ["All", "Male", "Female"]
# sel_gender = st.sidebar.selectbox("Gender", gender_opts)

# # Age filter
# age_opts = ["All"] + list(DIMENSIONS['age_level'].values())
# sel_age = st.sidebar.selectbox("Age Group", age_opts)

# # Shopping level
# shop_opts = ["All"] + list(DIMENSIONS['shopping_level'].values())
# sel_shop = st.sidebar.selectbox("Shopping Depth", shop_opts)

# # 应用 filter
# df_f = df.copy()
# if sel_gender != "All":
#     gender_code = 1 if sel_gender == "Male" else 2
#     df_f = df_f[df_f['final_gender_code'] == gender_code]
# if sel_age != "All":
#     age_code = [k for k, v in DIMENSIONS['age_level'].items() if v == sel_age][0]
#     df_f = df_f[df_f['age_level'] == age_code]
# if sel_shop != "All":
#     shop_code = [k for k, v in DIMENSIONS['shopping_level'].items() if v == sel_shop][0]
#     df_f = df_f[df_f['shopping_level'] == shop_code]

# # ============ 1. Core KPIs ============
# st.subheader("📊 Core KPIs")

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Impressions", f"{len(df_f):,}")
# col2.metric("Clicks", f"{df_f['clk'].sum():,}")
# col3.metric("CTR", f"{df_f['clk'].mean()*100:.2f}%")
# col4.metric("Unique Users", f"{df_f['user_id'].nunique():,}")

# if len(df_f) < len(df):
#     st.caption(f"Filtered view: {len(df_f):,} of {len(df):,} rows ({len(df_f)/len(df)*100:.1f}%)")

# st.markdown("---")

# # ============ 2. CTR Spread Summary(Tab 1 的核心 killer 图)============

# # ============ 2. CTR Spread Summary ============
# st.subheader("🎯 CTR Spread by User Dimension")
# st.caption("How much does CTR vary across each dimension?  \nLarger spread = stronger modeling signal.  \n**pp** = percentage points (absolute CTR difference between highest and lowest segment).")

# spread_data = []
# for dim, label_map in DIMENSIONS.items():
#     sub = df[[dim, 'clk']].dropna()
#     grp = sub.groupby(dim)['clk'].agg(['count', 'mean']).reset_index()
#     grp = grp[grp['count'] >= MIN_IMPRESSIONS]
#     if len(grp) > 0:
#         ctrs = grp['mean'] * 100
#         spread_data.append({
#             'dimension': NICE_NAMES.get(dim, dim),
#             'min_ctr': ctrs.min(),
#             'max_ctr': ctrs.max(),
#             'spread': ctrs.max() - ctrs.min()
#         })

# spread_df = pd.DataFrame(spread_data).sort_values('spread', ascending=True)

# fig, ax = plt.subplots(figsize=(10, 4.5))
# colors = ['#27ae60' if s > 0.3 else '#e67e22' if s > 0.15 else '#95a5a6'
#           for s in spread_df['spread']]
# bars = ax.barh(spread_df['dimension'], spread_df['spread'], color=colors, edgecolor='white')
# for i, (dim, spread) in enumerate(zip(spread_df['dimension'], spread_df['spread'])):
#     ax.text(spread + 0.01, i, f"{spread:.2f}pp", va='center', fontsize=10, fontweight='bold')
# ax.set_xlabel('CTR Spread (percentage points)', fontsize=11)
# ax.set_title('User Dimension Impact Ranking', fontsize=13, fontweight='bold', pad=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlim(0, max(spread_df['spread']) * 1.2)
# plt.tight_layout()
# st.pyplot(fig)
# plt.close(fig)

# # 加 legend 解释颜色含义
# col_a, col_b, col_c = st.columns(3)
# col_a.markdown("🟢 **Strong signal** (>0.30pp)")
# col_b.markdown("🟠 **Moderate** (0.15-0.30pp)")
# col_c.markdown("⚪ **Weak** (<0.15pp)")

# st.info("💡 **Modeling Implication**: Age Group (0.66pp) and Gender (0.39pp) show the strongest "
#         "CTR variation — these should be first-priority features. Shopping Depth and City Tier "
#         "add moderate signal. Low-spread dimensions risk adding noise without useful signal.")

# # ============ 3. Temporal Trends ============
# st.subheader("⏰ Temporal Trends")

# col_l, col_r = st.columns(2)

# # Hourly CTR
# # Hourly CTR
# with col_l:
#     hourly = df_f.groupby('hour').agg(
#         impressions=('clk', 'count'),
#         clicks=('clk', 'sum')
#     ).reset_index()
#     hourly['ctr'] = hourly['clicks'] / hourly['impressions'] * 100
    
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax2 = ax.twinx()
#     ax.bar(hourly['hour'], hourly['impressions'], color='steelblue',
#            alpha=0.3, label='Impressions')
#     ax2.plot(hourly['hour'], hourly['ctr'], color='#e74c3c',
#              marker='o', linewidth=2.5, markersize=6, label='CTR')
#     ax2.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
#                 alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
#     # 标注 golden hour(最高 CTR 的时段)
#     peak_row = hourly.loc[hourly['ctr'].idxmax()]
#     ax2.annotate(f"Peak: {peak_row['ctr']:.2f}%\n@ {int(peak_row['hour'])}:00",
#                  xy=(peak_row['hour'], peak_row['ctr']),
#                  xytext=(peak_row['hour']-3, peak_row['ctr']+0.3),
#                  fontsize=9, color='#c0392b',
#                  arrowprops=dict(arrowstyle='->', color='#c0392b', alpha=0.6))
    
#     ax.set_xlabel('Hour (Beijing time)', fontsize=10)
#     ax.set_ylabel('Impressions', color='steelblue', fontsize=10)
#     ax2.set_ylabel('CTR (%)', color='#e74c3c', fontsize=10)
#     ax2.set_ylim(4.0, max(hourly['ctr']) + 0.5)
#     ax.set_title('Hourly Pattern', fontsize=12, fontweight='bold')
#     ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
#     ax.spines['top'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# # Weekday vs Weekend
# # Weekday vs Weekend
# with col_r:
#     day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#     daily = df_f.groupby('day_of_week').agg(
#         impressions=('clk', 'count'),
#         clicks=('clk', 'sum')
#     ).reset_index()
#     daily['ctr'] = daily['clicks'] / daily['impressions'] * 100
#     daily['day_name'] = daily['day_of_week'].apply(lambda x: day_names[x])
    
#     fig, ax = plt.subplots(figsize=(8, 4))
#     colors_day = ['#3498db' if d < 5 else '#e67e22' for d in daily['day_of_week']]
#     bars = ax.bar(daily['day_name'], daily['ctr'], color=colors_day, edgecolor='white')
#     ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
#                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
#     # 柱子顶部加数值
#     for bar, val in zip(bars, daily['ctr']):
#         ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
#                 f'{val:.2f}', ha='center', fontsize=9)
    
#     ax.set_ylabel('CTR (%)', fontsize=10)
#     ax.set_ylim(4.5, max(daily['ctr']) + 0.3)
#     ax.set_title('CTR by Day of Week', fontsize=12, fontweight='bold')
#     ax.legend(fontsize=8, loc='lower right')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# # 在 Temporal 段最后加一个解读框
# st.info("💡 **Temporal Insight**: CTR remains remarkably stable across days and hours "
#         "(spread <0.4pp), suggesting that **time-of-day features alone offer limited signal**. "
#         "Impression volume peaks at 9-10 PM, but CTR does not follow — evening users browse "
#         "more but don't click proportionally more.")
# # ============ 4. CTR by Placement & Price ============
# # ============ 4. CTR by Placement & Price ============
# st.subheader("📍 Ad Placement & Price")

# col_l, col_r = st.columns(2)

# # pid
# with col_l:
#     pid_grp = df_f.groupby('pid').agg(
#         impressions=('clk', 'count'),
#         clicks=('clk', 'sum')
#     ).reset_index()
#     pid_grp['ctr'] = pid_grp['clicks'] / pid_grp['impressions'] * 100
    
#     fig, ax = plt.subplots(figsize=(8, 4))
#     bars = ax.bar(pid_grp['pid'], pid_grp['ctr'], 
#                    color=['#3498db', '#e67e22'], edgecolor='white')
#     ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
#                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
#     for bar, val, imp in zip(bars, pid_grp['ctr'], pid_grp['impressions']):
#         ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
#                 f'{val:.2f}%\n(n={imp:,})', ha='center', fontsize=9)
#     ax.set_ylabel('CTR (%)', fontsize=10)
#     ax.set_ylim(4.5, max(pid_grp['ctr']) + 0.6)
#     ax.set_title('CTR by Ad Placement', fontsize=12, fontweight='bold')
#     ax.tick_params(axis='x', rotation=15)
#     ax.legend(fontsize=8)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# # Price bucket
# with col_r:
#     price_grp = df_f.groupby('price_bucket', observed=True).agg(
#         impressions=('clk', 'count'),
#         clicks=('clk', 'sum')
#     ).reset_index()
#     price_grp['ctr'] = price_grp['clicks'] / price_grp['impressions'] * 100
    
#     fig, ax = plt.subplots(figsize=(8, 4))
#     # 渐变色:越便宜 CTR 越高,用暖色;越贵越冷色
#     colors_price = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
#     bars = ax.bar(price_grp['price_bucket'].astype(str), price_grp['ctr'], 
#                    color=colors_price, edgecolor='white')
#     ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
#                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
#     for bar, val in zip(bars, price_grp['ctr']):
#         ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
#                 f'{val:.2f}%', ha='center', fontsize=9)
#     ax.set_ylabel('CTR (%)', fontsize=10)
#     ax.set_xlabel('Price Range (CNY)', fontsize=10)
#     ax.set_ylim(4.0, max(price_grp['ctr']) + 0.5)
#     ax.set_title('CTR by Price Range (p99 capped)', fontsize=12, fontweight='bold')
#     ax.legend(fontsize=8)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# st.info("💡 **Finding**: Price shows a clear monotonic decrease — the 0-100 CNY sweet spot "
#         "is consistent across user segments. Ad placement difference is modest (~0.3pp), "
#         "suggesting placement may be less impactful than user or price characteristics.")


# # ============ 5. User Segment CTR ============
# def plot_segment(ax, dim, label_map, title):
#     grp = df_f[[dim, 'clk']].dropna().groupby(dim).agg(
#         impressions=('clk', 'count'),
#         clicks=('clk', 'sum')
#     ).reset_index()
#     grp['ctr'] = grp['clicks'] / grp['impressions'] * 100
#     grp['label'] = grp[dim].map(label_map)
#     grp = grp.dropna(subset=['label'])
#     grp['label'] = grp['label'].astype(str)
    
#     bars = ax.bar(grp['label'], grp['ctr'], color='#3498db', edgecolor='white')
#     ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
#                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
#     for bar, val in zip(bars, grp['ctr']):
#         ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
#                 f'{val:.2f}', ha='center', fontsize=9)
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_ylabel('CTR (%)', fontsize=10)
#     # Y 轴聚焦:从接近最小值开始,顶部留空间
#     y_min = min(grp['ctr'].min(), BASELINE_CTR) - 0.3
#     y_max = grp['ctr'].max() + 0.4
#     ax.set_ylim(max(0, y_min), y_max)
#     ax.legend(fontsize=8, loc='lower right')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#  # ============ 6. Profile Matching Signal(来自 2b 的巧思)============
# st.subheader("🔍 Missingness as a Signal")

# match_grp = df.groupby('is_profile_matched').agg(
#     impressions=('clk', 'count'),
#     clicks=('clk', 'sum')
# ).reset_index()
# match_grp['ctr'] = match_grp['clicks'] / match_grp['impressions'] * 100
# match_grp['label'] = match_grp['is_profile_matched'].map(
#     {0: 'Unmatched (profile missing)', 1: 'Matched'}
# )

# col_l, col_r = st.columns([2, 1])
# with col_l:
#     fig, ax = plt.subplots(figsize=(8, 3))
#     bars = ax.barh(match_grp['label'], match_grp['ctr'], color=['#e74c3c', '#2ecc71'])
#     ax.axvline(x=BASELINE_CTR, color='red', linestyle='--',
#                alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
#     for bar, val, imp in zip(bars, match_grp['ctr'], match_grp['impressions']):
#         ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
#                 f'{val:.2f}% (n={imp:,})', va='center', fontsize=10)
#     ax.set_xlabel('CTR (%)')
#     ax.set_title('CTR: Profile-matched vs Unmatched Users')
#     ax.legend(fontsize=8)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# with col_r:
#     st.info("💡 **Key Insight**: Profile-missing rows are **not noise** — they form a distinct "
#             "behavioral segment. We engineered `is_profile_matched` as a binary feature to capture "
#             "this signal, rather than discarding or imputing the information.\n\n"
#             "This is a **methodology improvement** over baseline approaches that simply fill NaN "
#             "with median/mode.")

# st.markdown("---")

# # ============ 关键发现总结 ============
# st.markdown("---")
# st.subheader("🎯 Key Findings & Modeling Recommendations")

# col_a, col_b = st.columns(2)

# with col_a:
#     st.markdown("""
#     **🔝 High-priority features**
#     - `age_level` (0.66pp spread)
#     - `final_gender_code` (0.39pp)
#     - `pvalue_level` (0.37pp)
#     - `price_bucket` (monotonic trend)
#     - `is_profile_matched` (novel)
#     """)

# with col_b:
#     st.markdown("""
#     **⚠️ Limited-signal features**
#     - `shopping_level` (0.20pp)
#     - `new_user_class_level` (0.20pp)
#     - `hour_of_day` (<0.5pp variation)
#     - Anonymous ID rankings (`cate_id`, `brand`) — interpretable only in aggregate
#     """)

# st.success("✅ **Recommendation for modeling**: Prioritize the 5 high-signal features above. "
#            "Explore `age × pvalue` and `shopping × pvalue` interactions (not shown here — "
#            "see Module 2b for interaction heatmaps with 1.55pp spread).")

# # ============ 7. Methodology Note ============
# with st.expander("📋 Methodology Note: Missing Value Handling"):
#     st.markdown("""
#     Different analyses use different missing-value strategies based on their goal:

#     - **Single-dimension CTR views**: Retain NaN as a visible category (BigQuery default)
#     - **Spread ranking & interaction heatmaps**: `dropna` + minimum sample threshold (n≥500)
#     - **Statistical tests**: `dropna` required for chi-square
#     - **Clustering (K-means)**: Median imputation (distance-based method cannot handle NaN)
#     - **LightGBM**: Native NaN handling
#     - **Novel**: `is_profile_matched` feature captures missingness as signal

#     **Price outliers**: Capped at p99 (≈5,900 CNY) to reduce skew. Top 1% represents
#     luxury segment and does not materially affect CTR patterns.
#     """)

# st.caption("📊 Data: Alibaba Taobao Advertising Dataset | Dashboard by Wanman Gao | "
#            "UCLA MDSH Capstone 2026")

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import duckdb  # type: ignore

# ============ 页面配置 ============
st.set_page_config(page_title="Alibaba CTR Dashboard", layout="wide")
st.markdown("## 🛍️ Alibaba CTR Prediction — Business Overview")

# ============ DuckDB 连接 ============
DATA_PATH = "data_full/*.parquet"

@st.cache_resource
def get_db():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW wide AS SELECT * FROM read_parquet('{DATA_PATH}')")
    return con

con = get_db()

def run_query(sql):
    return con.execute(sql).fetchdf()

# ============ CTR 置信区间计算函数 ============
def ctr_ci(clicks, impressions, z=1.96):
    """计算 CTR 的 95% 置信区间（Wilson score interval 简化版）"""
    p = clicks / impressions
    se = np.sqrt(p * (1 - p) / impressions)
    return p * 100, se * z * 100  # 返回 CTR% 和 误差范围%

# ============ 获取基础信息 ============
@st.cache_data(ttl=3600)
def get_baseline_stats():
    sql = """
    SELECT
        COUNT(*) AS total_rows,
        SUM(clk) AS total_clicks,
        AVG(clk) * 100 AS overall_ctr,
        COUNT(DISTINCT user_id) AS unique_users
    FROM wide
    """
    return run_query(sql).iloc[0]

baseline = get_baseline_stats()
BASELINE_CTR = round(baseline['overall_ctr'], 2)
TOTAL_ROWS = int(baseline['total_rows'])
TOTAL_USERS = int(baseline['unique_users'])
AVG_IMP_PER_USER = TOTAL_ROWS / TOTAL_USERS

st.caption(f"Based on full {TOTAL_ROWS:,}-row dataset · Overall CTR baseline = {BASELINE_CTR}%  \nUCLA MDSH Capstone 2026")

with st.expander("ℹ️ About this dashboard"):
    st.markdown(f"""
    This dashboard explores user behavior patterns on Alibaba's Taobao advertising dataset
    to inform CTR prediction modeling. Every chart is designed to answer: 
    *"What does this tell us for feature engineering?"*
    
    **How to use**: Apply filters in the left sidebar to slice by gender, age, or shopping depth.  
    **Data**: {TOTAL_ROWS:,} impressions · {TOTAL_USERS:,} unique users · 
    {AVG_IMP_PER_USER:.1f} avg impressions/user · 8-day window (May 6–13, 2017).  
    **Backend**: DuckDB scans Parquet files on disk — minimal memory usage.  
    **Error bars**: All CTR bars show 95% confidence intervals.
    """)

# ============ 常量 ============
DIMENSIONS = {
    'final_gender_code': {1: 'Male', 2: 'Female'},
    'age_level':         {1:'<18', 2:'18-24', 3:'25-29', 4:'30-34',
                          5:'35-39', 6:'40-49', 7:'50+'},
    'pvalue_level':      {1: 'Low', 2: 'Mid', 3: 'High'},
    'shopping_level':    {1: 'Shallow', 2: 'Moderate', 3: 'Deep'},
    'occupation':        {0: 'Non-Student', 1: 'Student'},
    'new_user_class_level': {1:'Tier-1', 2:'Tier-2', 3:'Tier-3', 4:'Rural/Other'}
}
MIN_IMPRESSIONS = 500
NICE_NAMES = {
    'age_level': 'Age Group',
    'final_gender_code': 'Gender',
    'pvalue_level': 'Consumption Level',
    'shopping_level': 'Shopping Depth',
    'new_user_class_level': 'City Tier',
    'occupation': 'Occupation'
}

# ============ 构建 WHERE 子句 ============
def build_where(sel_gender, sel_age, sel_shop):
    conditions = []
    if sel_gender != "All":
        code = 1 if sel_gender == "Male" else 2
        conditions.append(f"final_gender_code = {code}")
    if sel_age != "All":
        code = [k for k, v in DIMENSIONS['age_level'].items() if v == sel_age][0]
        conditions.append(f"age_level = {code}")
    if sel_shop != "All":
        code = [k for k, v in DIMENSIONS['shopping_level'].items() if v == sel_shop][0]
        conditions.append(f"shopping_level = {code}")
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""

# ============ Sidebar Filters ============
st.sidebar.header("🔍 Filters")
st.sidebar.caption("Filters apply to all charts below.")

sel_gender = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
sel_age = st.sidebar.selectbox("Age Group", ["All"] + list(DIMENSIONS['age_level'].values()))
sel_shop = st.sidebar.selectbox("Shopping Depth", ["All"] + list(DIMENSIONS['shopping_level'].values()))

WHERE = build_where(sel_gender, sel_age, sel_shop)

# ============ 1. Core KPIs (加了 avg impressions per user) ============
st.subheader("📊 Core KPIs")

@st.cache_data(ttl=3600)
def get_kpis(where_clause):
    sql = f"""
    SELECT
        COUNT(*) AS impressions,
        SUM(clk) AS clicks,
        AVG(clk) * 100 AS ctr,
        COUNT(DISTINCT user_id) AS unique_users
    FROM wide
    {where_clause}
    """
    return run_query(sql).iloc[0]

kpis = get_kpis(WHERE)
avg_imp = int(kpis['impressions']) / max(int(kpis['unique_users']), 1)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Impressions", f"{int(kpis['impressions']):,}")
col2.metric("Total Clicks", f"{int(kpis['clicks']):,}")
col3.metric("CTR", f"{kpis['ctr']:.2f}%")
col4.metric("Unique Users", f"{int(kpis['unique_users']):,}")
col5.metric("Avg Imp / User", f"{avg_imp:.1f}")

if WHERE:
    st.caption(f"Filtered view: {int(kpis['impressions']):,} of {TOTAL_ROWS:,} rows "
               f"({int(kpis['impressions'])/TOTAL_ROWS*100:.1f}%)")

st.markdown("---")

# ============ 2. User Segment Breakdown ============
st.subheader("👥 User Segment Breakdown")
st.caption("For each user feature, how are users and impressions distributed across segments?  \n"
           "Each chart shows: **user count** (bars, left axis), **CTR with 95% CI** (line, right axis).  \n"
           "Numbers above bars = impressions count · avg impressions per user.")
 
@st.cache_data(ttl=3600)
def get_segment_data(dim, where_clause):
    filter_clause = where_clause.replace("WHERE", "AND") if where_clause else ""
    sql = f"""
    SELECT
        {dim} AS segment,
        COUNT(*) AS impressions,
        SUM(clk) AS clicks,
        COUNT(DISTINCT user_id) AS users,
        AVG(clk) * 100 AS ctr
    FROM wide
    WHERE {dim} IS NOT NULL {filter_clause}
    GROUP BY {dim}
    HAVING COUNT(*) >= {MIN_IMPRESSIONS}
    ORDER BY {dim}
    """
    return run_query(sql)
 
def plot_segment_breakdown(dim, label_map, title, where_clause):
    """每个特征画一个双轴图：bars = users, line = CTR ± CI"""
    grp = get_segment_data(dim, where_clause)
    if len(grp) == 0:
        st.warning(f"No data for {title}")
        return
    
    grp['label'] = grp['segment'].apply(lambda x: label_map.get(int(x), str(x)))
    grp['avg_imp_per_user'] = (grp['impressions'] / grp['users']).round(1)
    
    # 计算 95% CI
    ctrs = []
    cis = []
    for _, row in grp.iterrows():
        c, ci = ctr_ci(row['clicks'], row['impressions'])
        ctrs.append(c)
        cis.append(ci)
    grp['ctr_calc'] = ctrs
    grp['ci'] = cis
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    
    x = np.arange(len(grp))
    bar_width = 0.5
    
    # 柱子 = users count
    bars = ax.bar(x, grp['users'], width=bar_width, color='#3498db', alpha=0.6,
                  edgecolor='white', label='Users')
    
    # 柱子上方标注 impressions 和 avg imp/user
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(i, row['users'] + grp['users'].max() * 0.02,
                f"{int(row['impressions']):,} imp\n{row['avg_imp_per_user']} imp/user",
                ha='center', fontsize=7, color='#555')
    
    # 折线 = CTR with error bars
    ax2.errorbar(x, grp['ctr'], yerr=grp['ci'],
                 color='#e74c3c', marker='o', linewidth=2, markersize=6,
                 capsize=4, capthick=1.5, label='CTR ± 95% CI', zorder=5)
    ax2.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
    # CTR 数值标注在点旁边
    for i, row in grp.reset_index(drop=True).iterrows():
        ax2.annotate(f"{row['ctr']:.2f}%", xy=(i, row['ctr']),
                     xytext=(5, 8), textcoords='offset points',
                     fontsize=8, color='#c0392b', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(grp['label'], fontsize=9)
    ax.set_ylabel('Number of Users', color='#3498db', fontsize=10)
    ax2.set_ylabel('CTR (%)', color='#e74c3c', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Y轴范围
    ax2.set_ylim(max(0, grp['ctr'].min() - 0.5), grp['ctr'].max() + 0.5)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
 
# 2x3 网格布局，6 个特征各画一个图
row1_l, row1_r = st.columns(2)
with row1_l:
    plot_segment_breakdown('age_level', DIMENSIONS['age_level'],
                           'Age Group: Users & CTR', WHERE)
with row1_r:
    plot_segment_breakdown('final_gender_code', DIMENSIONS['final_gender_code'],
                           'Gender: Users & CTR', WHERE)
 
row2_l, row2_r = st.columns(2)
with row2_l:
    plot_segment_breakdown('pvalue_level', DIMENSIONS['pvalue_level'],
                           'Consumption Level: Users & CTR', WHERE)
with row2_r:
    plot_segment_breakdown('shopping_level', DIMENSIONS['shopping_level'],
                           'Shopping Depth: Users & CTR', WHERE)
 
row3_l, row3_r = st.columns(2)
with row3_l:
    plot_segment_breakdown('new_user_class_level', DIMENSIONS['new_user_class_level'],
                           'City Tier: Users & CTR', WHERE)
with row3_r:
    plot_segment_breakdown('occupation', DIMENSIONS['occupation'],
                           'Occupation: Users & CTR', WHERE)
 
st.info("💡 **Key Observations**: Each chart reveals both the **size** (how many users/impressions) "
        "and **behavior** (CTR) of each segment. Features where CTR visibly differs across segments "
        "with large user populations are the strongest candidates for modeling.")
# ============ 3. Temporal Trends — Calendar Dates (删掉 weekly) ============
st.subheader("⏰ Temporal Trends")

col_l, col_r = st.columns(2)

# 左图: Hourly CTR with error bars
with col_l:
    @st.cache_data(ttl=3600)
    def get_hourly(where_clause):
        filter_clause = where_clause.replace("WHERE", "AND") if where_clause else ""
        sql = f"""
        SELECT
            EXTRACT(HOUR FROM to_timestamp(time_stamp) AT TIME ZONE 'Asia/Shanghai') AS hour,
            COUNT(*) AS impressions,
            SUM(clk) AS clicks,
            COUNT(DISTINCT user_id) AS users,
            AVG(clk) * 100 AS ctr
        FROM wide
        WHERE TRUE {filter_clause}
        GROUP BY hour
        ORDER BY hour
        """
        return run_query(sql)
    
    hourly = get_hourly(WHERE)
    _, hourly_ci = zip(*[ctr_ci(c, n) for c, n in zip(hourly['clicks'], hourly['impressions'])])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.bar(hourly['hour'], hourly['impressions'], color='steelblue',
           alpha=0.3, label='Impressions')
    ax2.errorbar(hourly['hour'], hourly['ctr'], yerr=hourly_ci,
                 color='#e74c3c', marker='o', linewidth=2, markersize=5,
                 capsize=3, capthick=1, label='CTR ± 95% CI')
    ax2.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
    peak_row = hourly.loc[hourly['ctr'].idxmax()]
    ax2.annotate(f"Peak: {peak_row['ctr']:.2f}%\n@ {int(peak_row['hour'])}:00",
                 xy=(peak_row['hour'], peak_row['ctr']),
                 xytext=(peak_row['hour']-3, peak_row['ctr']+0.3),
                 fontsize=9, color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', alpha=0.6))
    
    ax.set_xlabel('Hour (Beijing time)', fontsize=10)
    ax.set_ylabel('Impressions', color='steelblue', fontsize=10)
    ax2.set_ylabel('CTR (%)', color='#e74c3c', fontsize=10)
    ax2.set_ylim(4.0, max(hourly['ctr']) + 0.5)
    ax.set_title('Hourly CTR Pattern (with 95% CI)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 右图: Daily CTR by Calendar Date (替换 weekday 图)
with col_r:
    @st.cache_data(ttl=3600)
    def get_daily_calendar(where_clause):
        filter_clause = where_clause.replace("WHERE", "AND") if where_clause else ""
        sql = f"""
        SELECT
            CAST(to_timestamp(time_stamp) AT TIME ZONE 'Asia/Shanghai' AS DATE) AS cal_date,
            COUNT(*) AS impressions,
            SUM(clk) AS clicks,
            COUNT(DISTINCT user_id) AS users,
            AVG(clk) * 100 AS ctr
        FROM wide
        WHERE TRUE {filter_clause}
        GROUP BY cal_date
        ORDER BY cal_date
        """
        return run_query(sql)
    
    daily = get_daily_calendar(WHERE)
    _, daily_ci = zip(*[ctr_ci(c, n) for c, n in zip(daily['clicks'], daily['impressions'])])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.bar(daily['cal_date'], daily['impressions'], color='steelblue',
           alpha=0.3, width=0.8, label='Impressions')
    ax2.errorbar(daily['cal_date'], daily['ctr'], yerr=daily_ci,
                 color='#e74c3c', marker='o', linewidth=2, markersize=5,
                 capsize=3, capthick=1, label='CTR ± 95% CI')
    ax2.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
    # 每个点标注 users 数
    for _, row in daily.iterrows():
        ax2.annotate(f"{int(row['users']):,}u",
                     xy=(row['cal_date'], row['ctr']),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=7, ha='center', color='#555')
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Impressions', color='steelblue', fontsize=10)
    ax2.set_ylabel('CTR (%)', color='#e74c3c', fontsize=10)
    ax2.set_ylim(4.0, max(daily['ctr']) + 0.6)
    ax.set_title('Daily CTR by Calendar Date (with 95% CI)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='x', rotation=30)
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Temporal 表格
with st.expander("📋 View daily counts"):
    daily_display = daily.copy()
    daily_display['Avg Imp/User'] = (daily_display['impressions'] / daily_display['users']).round(1)
    daily_display.columns = ['Date', 'Impressions', 'Clicks', 'Users', 'CTR (%)', 'Avg Imp/User']
    st.dataframe(daily_display, use_container_width=True, hide_index=True)

st.info("💡 **Temporal Insight**: CTR remains remarkably stable across dates "
        "(spread <0.4pp), suggesting that **time features alone offer limited signal**. "
        "Impression volume peaks in evening hours, but CTR does not follow — evening users browse "
        "more but don't click proportionally more.")

# ============ 4. CTR by Placement & Price (with error bars & counts) ============
st.subheader("📍 Ad Placement & Price")

col_l, col_r = st.columns(2)

# PID with error bars
with col_l:
    @st.cache_data(ttl=3600)
    def get_pid_ctr(where_clause):
        filter_clause = where_clause.replace("WHERE", "AND") if where_clause else ""
        sql = f"""
        SELECT
            pid,
            COUNT(*) AS impressions,
            SUM(clk) AS clicks,
            COUNT(DISTINCT user_id) AS users,
            AVG(clk) * 100 AS ctr
        FROM wide
        WHERE TRUE {filter_clause}
        GROUP BY pid
        ORDER BY pid
        """
        return run_query(sql)
    
    pid_grp = get_pid_ctr(WHERE)
    _, pid_ci = zip(*[ctr_ci(c, n) for c, n in zip(pid_grp['clicks'], pid_grp['impressions'])])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(pid_grp['pid'], pid_grp['ctr'], yerr=pid_ci,
                   color=['#3498db', '#e67e22'], edgecolor='white',
                   capsize=5, error_kw={'capthick': 1.5})
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    for bar, val, imp, usr in zip(bars, pid_grp['ctr'], pid_grp['impressions'], pid_grp['users']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f'{val:.2f}%\n{int(imp):,} imp\n{int(usr):,} users',
                ha='center', fontsize=8)
    ax.set_ylabel('CTR (%)', fontsize=10)
    ax.set_ylim(4.0, max(pid_grp['ctr']) + 1.0)
    ax.set_title('CTR by Ad Placement (± 95% CI)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Price bucket with error bars
with col_r:
    @st.cache_data(ttl=3600)
    def get_price_ctr(where_clause):
        filter_clause = where_clause.replace("WHERE", "AND") if where_clause else ""
        sql = f"""
        WITH p99 AS (
            SELECT quantile_cont(price, 0.99) AS p99_val FROM wide
        )
        SELECT
            CASE
                WHEN price <= 100 THEN '0-100'
                WHEN price <= 300 THEN '100-300'
                WHEN price <= 600 THEN '300-600'
                WHEN price <= 1320 THEN '600-1320'
                ELSE '1320-5900'
            END AS price_bucket,
            COUNT(*) AS impressions,
            SUM(clk) AS clicks,
            COUNT(DISTINCT user_id) AS users,
            AVG(clk) * 100 AS ctr
        FROM wide, p99
        WHERE price <= p99.p99_val
        {filter_clause.replace("WHERE", "AND")}
        GROUP BY price_bucket
        ORDER BY MIN(price)
        """
        return run_query(sql)
    
    price_grp = get_price_ctr(WHERE)
    _, price_ci = zip(*[ctr_ci(c, n) for c, n in zip(price_grp['clicks'], price_grp['impressions'])])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_price = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    n_bars = len(price_grp)
    bars = ax.bar(price_grp['price_bucket'].astype(str), price_grp['ctr'],
                   yerr=price_ci[:n_bars], color=colors_price[:n_bars], edgecolor='white',
                   capsize=5, error_kw={'capthick': 1.5})
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    for bar, val, imp, usr in zip(bars, price_grp['ctr'], price_grp['impressions'], price_grp['users']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f'{val:.2f}%\n{int(imp):,} imp\n{int(usr):,} users',
                ha='center', fontsize=7)
    ax.set_ylabel('CTR (%)', fontsize=10)
    ax.set_xlabel('Price Range (CNY)', fontsize=10)
    ax.set_ylim(3.5, max(price_grp['ctr']) + 1.0)
    ax.set_title('CTR by Price Range (± 95% CI)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.info("💡 **Finding**: Price shows a clear monotonic decrease — the 0-100 CNY sweet spot "
        "is consistent across user segments. Ad placement difference is modest (~0.3pp), "
        "suggesting placement may be less impactful than user or price characteristics.")

# ============ 5. Bubble/Scatter Plots — Quadrant Analysis ============
st.subheader("🫧 Categorical Data: Impression Volume vs CTR (Quadrant Analysis)")
st.caption("Each bubble = one category/brand/advertiser. Bubble size = number of unique users.  \n"
           "Quadrants help identify high-value segments for targeting.")

@st.cache_data(ttl=3600)
def get_bubble_data(dim_col, top_n=50):
    sql = f"""
    SELECT
        CAST({dim_col} AS VARCHAR) AS segment,
        COUNT(*) AS impressions,
        SUM(clk) AS clicks,
        COUNT(DISTINCT user_id) AS users,
        AVG(clk) * 100 AS ctr
    FROM wide
    WHERE {dim_col} IS NOT NULL
    GROUP BY segment
    HAVING COUNT(*) >= 1000
    ORDER BY impressions DESC
    LIMIT {top_n}
    """
    return run_query(sql)

bubble_tab = st.selectbox("Select dimension", ["cate_id (Category)", "brand (Brand)", "adgroup_id (Advertiser)"])
dim_col = bubble_tab.split(" ")[0]

bubble_df = get_bubble_data(dim_col, top_n=80)

if len(bubble_df) > 0:
    median_imp = bubble_df['impressions'].median()
    median_ctr = bubble_df['ctr'].median()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 气泡大小映射
    size_scale = bubble_df['users'] / bubble_df['users'].max() * 500 + 20
    
    # 四象限颜色
    colors_quad = []
    for _, row in bubble_df.iterrows():
        if row['impressions'] >= median_imp and row['ctr'] >= median_ctr:
            colors_quad.append('#27ae60')  # 高曝光 高CTR = 最优
        elif row['impressions'] < median_imp and row['ctr'] >= median_ctr:
            colors_quad.append('#3498db')  # 低曝光 高CTR = 潜力股
        elif row['impressions'] >= median_imp and row['ctr'] < median_ctr:
            colors_quad.append('#e67e22')  # 高曝光 低CTR = 需优化
        else:
            colors_quad.append('#95a5a6')  # 低曝光 低CTR = 低优先级
    
    scatter = ax.scatter(bubble_df['impressions'], bubble_df['ctr'],
                         s=size_scale, c=colors_quad, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # 画四象限线
    ax.axhline(y=median_ctr, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=median_imp, color='gray', linestyle='--', alpha=0.5)
    
    # 四象限标签
    ax.text(bubble_df['impressions'].max() * 0.95, bubble_df['ctr'].max() * 0.97,
            '★ High Imp + High CTR\n(Top performers)', fontsize=9, color='#27ae60',
            ha='right', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(bubble_df['impressions'].min() * 1.5, bubble_df['ctr'].max() * 0.97,
            '🔍 Low Imp + High CTR\n(Hidden gems)', fontsize=9, color='#3498db',
            ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(bubble_df['impressions'].max() * 0.95, bubble_df['ctr'].min() * 1.02,
            '⚠️ High Imp + Low CTR\n(Needs optimization)', fontsize=9, color='#e67e22',
            ha='right', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(bubble_df['impressions'].min() * 1.5, bubble_df['ctr'].min() * 1.02,
            '↓ Low Imp + Low CTR\n(Low priority)', fontsize=9, color='#95a5a6',
            ha='left', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Impressions (log scale)', fontsize=11)
    ax.set_ylabel('CTR (%)', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(f'Top {len(bubble_df)} {dim_col}s: Impression Volume vs CTR',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 四象限统计表
    bubble_df['quadrant'] = bubble_df.apply(lambda r:
        '★ High Imp + High CTR' if r['impressions'] >= median_imp and r['ctr'] >= median_ctr else
        '🔍 Low Imp + High CTR' if r['impressions'] < median_imp and r['ctr'] >= median_ctr else
        '⚠️ High Imp + Low CTR' if r['impressions'] >= median_imp and r['ctr'] < median_ctr else
        '↓ Low priority', axis=1)
    
    quad_summary = bubble_df.groupby('quadrant').agg(
        count=('segment', 'count'),
        total_impressions=('impressions', 'sum'),
        total_users=('users', 'sum'),
        avg_ctr=('ctr', 'mean')
    ).reset_index()
    quad_summary['avg_ctr'] = quad_summary['avg_ctr'].round(2)
    quad_summary.columns = ['Quadrant', 'Count', 'Total Impressions', 'Total Users', 'Avg CTR (%)']
    
    with st.expander("📋 Quadrant summary table"):
        st.dataframe(quad_summary, use_container_width=True, hide_index=True)
    
    st.info(f"💡 **Business Insight**: The quadrant view reveals that among top {len(bubble_df)} "
            f"{dim_col}s, the 'Hidden gems' (high CTR but low impression volume) represent "
            "untapped opportunities — increasing their ad exposure could improve overall CTR performance.")

# ============ 6. Profile Matching Signal (with counts & error bars) ============
st.subheader("🔍 Missingness as a Signal")

@st.cache_data(ttl=3600)
def get_profile_match():
    sql = """
    SELECT
        CASE WHEN age_level IS NOT NULL THEN 1 ELSE 0 END AS is_profile_matched,
        COUNT(*) AS impressions,
        SUM(clk) AS clicks,
        COUNT(DISTINCT user_id) AS users,
        AVG(clk) * 100 AS ctr
    FROM wide
    GROUP BY is_profile_matched
    """
    return run_query(sql)

match_grp = get_profile_match()
match_grp['label'] = match_grp['is_profile_matched'].map(
    {0: 'Unmatched (profile missing)', 1: 'Matched'}
)
_, match_ci = zip(*[ctr_ci(c, n) for c, n in zip(match_grp['clicks'], match_grp['impressions'])])

col_l, col_r = st.columns([2, 1])
with col_l:
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(match_grp['label'], match_grp['ctr'], xerr=match_ci,
                   color=['#e74c3c', '#2ecc71'], capsize=5, error_kw={'capthick': 1.5})
    ax.axvline(x=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    for bar, val, imp, usr in zip(bars, match_grp['ctr'], match_grp['impressions'], match_grp['users']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%  |  {int(imp):,} imp · {int(usr):,} users',
                va='center', fontsize=9)
    ax.set_xlabel('CTR (%)')
    ax.set_title('CTR: Profile-matched vs Unmatched (± 95% CI)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_r:
    st.info("💡 **Key Insight**: Profile-missing rows are **not noise** — they form a distinct "
            "behavioral segment. We engineered `is_profile_matched` as a binary feature to capture "
            "this signal, rather than discarding or imputing the information.\n\n"
            "This is a **methodology improvement** over baseline approaches that simply fill NaN "
            "with median/mode.")

st.markdown("---")

# ============ 7. Key Findings ============
st.markdown("---")
st.subheader("🎯 Key Findings & Modeling Recommendations")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **🔝 High-priority features**
    - `age_level` — strongest CTR spread
    - `final_gender_code` — significant variation
    - `pvalue_level` — consumption-level signal
    - `price_bucket` — monotonic trend
    - `is_profile_matched` — novel missingness feature
    """)

with col_b:
    st.markdown("""
    **⚠️ Limited-signal features**
    - `shopping_level` — modest spread
    - `new_user_class_level` — modest spread
    - `hour_of_day` — <0.5pp variation
    - Anonymous ID rankings (`cate_id`, `brand`) — see quadrant analysis above
    """)

st.success("✅ **Recommendation for modeling**: Prioritize the 5 high-signal features above. "
           "Use the quadrant analysis to identify high-value categories and brands for targeted "
           "feature engineering. Explore `age × pvalue` and `shopping × pvalue` interactions.")

# ============ 8. Methodology Note ============
with st.expander("📋 Methodology Note"):
    st.markdown(f"""
    **Absolute Counts**: All visualizations now display total impressions, unique user counts, 
    and average impressions per user alongside CTR rates, per reviewer feedback.
    
    **Error Bars**: 95% confidence intervals computed using normal approximation to binomial 
    (SE = √(p(1-p)/n)), appropriate given the large sample sizes (n > 500 per segment).
    
    **Quadrant Analysis**: Categorical dimensions (brand, category, advertiser) visualized as 
    bubble plots with median-based quadrant segmentation. Bubble size = unique user count.
    
    **Missing Value Handling**:
    - Spread ranking: `dropna` + minimum sample threshold (n≥500)
    - Novel: `is_profile_matched` feature captures missingness as signal

    **Price outliers**: Capped at p99 (≈5,900 CNY) to reduce skew.
    
    **Backend**: DuckDB scans {TOTAL_ROWS:,} rows from Parquet files on disk without loading 
    all data into memory.
    """)

st.caption(f"📊 Data: Alibaba Taobao Advertising Dataset ({TOTAL_ROWS:,} rows) | "
           f"Dashboard by Wanman Gao | UCLA MDSH Capstone 2026")