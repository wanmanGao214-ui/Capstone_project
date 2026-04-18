import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# ============ 页面配置 ============
st.set_page_config(page_title="Alibaba CTR Dashboard", layout="wide")
st.markdown("## 🛍️ Alibaba CTR Prediction — Business Overview")
st.caption("Based on full 1M-row sample · Overall CTR baseline = 5.16% · UCLA MDSH Capstone 2026")

with st.expander("ℹ️ About this dashboard"):
    st.markdown("""
    This dashboard explores user behavior patterns on Alibaba's Taobao advertising dataset
    to inform CTR prediction modeling. Every chart is designed to answer: 
    *"What does this tell us for feature engineering?"*
    
    **How to use**: Apply filters in the left sidebar to slice by gender, age, or shopping depth.  
    **Data**: 1,023,313 impressions · 408,578 unique users · 8-day window (May 6–13, 2017).
    """)

# ============ 常量:label 映射(来自 2b)============
DIMENSIONS = {
    'final_gender_code': {1: 'Male', 2: 'Female'},
    'age_level':         {1:'<18', 2:'18-24', 3:'25-29', 4:'30-34',
                          5:'35-39', 6:'40-49', 7:'50+'},
    'pvalue_level':      {1: 'Low', 2: 'Mid', 3: 'High'},
    'shopping_level':    {1: 'Shallow', 2: 'Moderate', 3: 'Deep'},
    'occupation':        {0: 'Non-Student', 1: 'Student'},
    'new_user_class_level': {1:'Tier-1', 2:'Tier-2', 3:'Tier-3', 4:'Rural/Other'}
}
MIN_IMPRESSIONS = 500  # 来自 2b,小样本过滤阈值
BASELINE_CTR = 5.16    # 来自 notebook 的整体 CTR
NICE_NAMES = {
    'age_level': 'Age Group',
    'final_gender_code': 'Gender',
    'pvalue_level': 'Consumption Level',
    'shopping_level': 'Shopping Depth',
    'new_user_class_level': 'City Tier',
    'occupation': 'Occupation'
} #让读者更好理解
# ============ 数据加载 ============
@st.cache_data
def load_data():
    df = pd.read_parquet("data/ctr_full.parquet")
    

# 时间处理:从 time_stamp (UNIX 秒) 构造
    df['event_time'] = pd.to_datetime(df['time_stamp'], unit='s', utc=True)
    df['event_time_beijing'] = df['event_time'].dt.tz_convert('Asia/Shanghai')
    df['hour'] = df['event_time_beijing'].dt.hour
    df['day_of_week'] = df['event_time_beijing'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Price 处理(来自 2c)
    p99 = df['price'].quantile(0.99)
    df['price_capped'] = df['price'].clip(upper=p99)
    df['price_bucket'] = pd.cut(
        df['price_capped'],
        bins=[0, 100, 300, 600, 1320, p99 + 1],
        labels=['0-100', '100-300', '300-600', '600-1320', '1320-5900']
    )
    
    # Profile matched flag(来自 2b 的巧思)
    df['is_profile_matched'] = df['age_level'].notna().astype(int)
    
    return df

df = load_data()

# ============ Sidebar Filters ============
st.sidebar.header("🔍 Filters")
st.sidebar.caption("Filters apply to all charts below.")

# Gender filter
gender_opts = ["All", "Male", "Female"]
sel_gender = st.sidebar.selectbox("Gender", gender_opts)

# Age filter
age_opts = ["All"] + list(DIMENSIONS['age_level'].values())
sel_age = st.sidebar.selectbox("Age Group", age_opts)

# Shopping level
shop_opts = ["All"] + list(DIMENSIONS['shopping_level'].values())
sel_shop = st.sidebar.selectbox("Shopping Depth", shop_opts)

# 应用 filter
df_f = df.copy()
if sel_gender != "All":
    gender_code = 1 if sel_gender == "Male" else 2
    df_f = df_f[df_f['final_gender_code'] == gender_code]
if sel_age != "All":
    age_code = [k for k, v in DIMENSIONS['age_level'].items() if v == sel_age][0]
    df_f = df_f[df_f['age_level'] == age_code]
if sel_shop != "All":
    shop_code = [k for k, v in DIMENSIONS['shopping_level'].items() if v == sel_shop][0]
    df_f = df_f[df_f['shopping_level'] == shop_code]

# ============ 1. Core KPIs ============
st.subheader("📊 Core KPIs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Impressions", f"{len(df_f):,}")
col2.metric("Clicks", f"{df_f['clk'].sum():,}")
col3.metric("CTR", f"{df_f['clk'].mean()*100:.2f}%")
col4.metric("Unique Users", f"{df_f['user_id'].nunique():,}")

if len(df_f) < len(df):
    st.caption(f"Filtered view: {len(df_f):,} of {len(df):,} rows ({len(df_f)/len(df)*100:.1f}%)")

st.markdown("---")

# ============ 2. CTR Spread Summary(Tab 1 的核心 killer 图)============

# ============ 2. CTR Spread Summary ============
st.subheader("🎯 CTR Spread by User Dimension")
st.caption("How much does CTR vary across each dimension? Larger spread = stronger modeling signal."
           "**pp** = percentage points (absolute CTR difference between highest and lowest segment).")

spread_data = []
for dim, label_map in DIMENSIONS.items():
    sub = df[[dim, 'clk']].dropna()
    grp = sub.groupby(dim)['clk'].agg(['count', 'mean']).reset_index()
    grp = grp[grp['count'] >= MIN_IMPRESSIONS]
    if len(grp) > 0:
        ctrs = grp['mean'] * 100
        spread_data.append({
            'dimension': NICE_NAMES.get(dim, dim),
            'min_ctr': ctrs.min(),
            'max_ctr': ctrs.max(),
            'spread': ctrs.max() - ctrs.min()
        })

spread_df = pd.DataFrame(spread_data).sort_values('spread', ascending=True)

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = ['#27ae60' if s > 0.3 else '#e67e22' if s > 0.15 else '#95a5a6'
          for s in spread_df['spread']]
bars = ax.barh(spread_df['dimension'], spread_df['spread'], color=colors, edgecolor='white')
for i, (dim, spread) in enumerate(zip(spread_df['dimension'], spread_df['spread'])):
    ax.text(spread + 0.01, i, f"{spread:.2f}pp", va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('CTR Spread (percentage points)', fontsize=11)
ax.set_title('User Dimension Impact Ranking', fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(spread_df['spread']) * 1.2)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# 加 legend 解释颜色含义
col_a, col_b, col_c = st.columns(3)
col_a.markdown("🟢 **Strong signal** (>0.30pp)")
col_b.markdown("🟠 **Moderate** (0.15-0.30pp)")
col_c.markdown("⚪ **Weak** (<0.15pp)")

st.info("💡 **Modeling Implication**: Age Group (0.66pp) and Gender (0.39pp) show the strongest "
        "CTR variation — these should be first-priority features. Shopping Depth and City Tier "
        "add moderate signal. Low-spread dimensions risk adding noise without useful signal.")

# ============ 3. Temporal Trends ============
st.subheader("⏰ Temporal Trends")

col_l, col_r = st.columns(2)

# Hourly CTR
# Hourly CTR
with col_l:
    hourly = df_f.groupby('hour').agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    hourly['ctr'] = hourly['clicks'] / hourly['impressions'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.bar(hourly['hour'], hourly['impressions'], color='steelblue',
           alpha=0.3, label='Impressions')
    ax2.plot(hourly['hour'], hourly['ctr'], color='#e74c3c',
             marker='o', linewidth=2.5, markersize=6, label='CTR')
    ax2.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
                alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
    # 标注 golden hour(最高 CTR 的时段)
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
    ax.set_title('Hourly Pattern', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Weekday vs Weekend
# Weekday vs Weekend
with col_r:
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = df_f.groupby('day_of_week').agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    daily['ctr'] = daily['clicks'] / daily['impressions'] * 100
    daily['day_name'] = daily['day_of_week'].apply(lambda x: day_names[x])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_day = ['#3498db' if d < 5 else '#e67e22' for d in daily['day_of_week']]
    bars = ax.bar(daily['day_name'], daily['ctr'], color=colors_day, edgecolor='white')
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    
    # 柱子顶部加数值
    for bar, val in zip(bars, daily['ctr']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)
    
    ax.set_ylabel('CTR (%)', fontsize=10)
    ax.set_ylim(4.5, max(daily['ctr']) + 0.3)
    ax.set_title('CTR by Day of Week', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 在 Temporal 段最后加一个解读框
st.info("💡 **Temporal Insight**: CTR remains remarkably stable across days and hours "
        "(spread <0.4pp), suggesting that **time-of-day features alone offer limited signal**. "
        "Impression volume peaks at 9-10 PM, but CTR does not follow — evening users browse "
        "more but don't click proportionally more.")
# ============ 4. CTR by Placement & Price ============
# ============ 4. CTR by Placement & Price ============
st.subheader("📍 Ad Placement & Price")

col_l, col_r = st.columns(2)

# pid
with col_l:
    pid_grp = df_f.groupby('pid').agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    pid_grp['ctr'] = pid_grp['clicks'] / pid_grp['impressions'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(pid_grp['pid'], pid_grp['ctr'], 
                   color=['#3498db', '#e67e22'], edgecolor='white')
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    for bar, val, imp in zip(bars, pid_grp['ctr'], pid_grp['impressions']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}%\n(n={imp:,})', ha='center', fontsize=9)
    ax.set_ylabel('CTR (%)', fontsize=10)
    ax.set_ylim(4.5, max(pid_grp['ctr']) + 0.6)
    ax.set_title('CTR by Ad Placement', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Price bucket
with col_r:
    price_grp = df_f.groupby('price_bucket', observed=True).agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    price_grp['ctr'] = price_grp['clicks'] / price_grp['impressions'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 4))
    # 渐变色:越便宜 CTR 越高,用暖色;越贵越冷色
    colors_price = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax.bar(price_grp['price_bucket'].astype(str), price_grp['ctr'], 
                   color=colors_price, edgecolor='white')
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    for bar, val in zip(bars, price_grp['ctr']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                f'{val:.2f}%', ha='center', fontsize=9)
    ax.set_ylabel('CTR (%)', fontsize=10)
    ax.set_xlabel('Price Range (CNY)', fontsize=10)
    ax.set_ylim(4.0, max(price_grp['ctr']) + 0.5)
    ax.set_title('CTR by Price Range (p99 capped)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.info("💡 **Finding**: Price shows a clear monotonic decrease — the 0-100 CNY sweet spot "
        "is consistent across user segments. Ad placement difference is modest (~0.3pp), "
        "suggesting placement may be less impactful than user or price characteristics.")


# ============ 5. User Segment CTR ============
def plot_segment(ax, dim, label_map, title):
    grp = df_f[[dim, 'clk']].dropna().groupby(dim).agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    grp['ctr'] = grp['clicks'] / grp['impressions'] * 100
    grp['label'] = grp[dim].map(label_map)
    grp = grp.dropna(subset=['label'])
    grp['label'] = grp['label'].astype(str)
    
    bars = ax.bar(grp['label'], grp['ctr'], color='#3498db', edgecolor='white')
    ax.axhline(y=BASELINE_CTR, color='gray', linestyle='--',
               alpha=0.6, label=f'Baseline {BASELINE_CTR}%')
    for bar, val in zip(bars, grp['ctr']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                f'{val:.2f}', ha='center', fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('CTR (%)', fontsize=10)
    # Y 轴聚焦:从接近最小值开始,顶部留空间
    y_min = min(grp['ctr'].min(), BASELINE_CTR) - 0.3
    y_max = grp['ctr'].max() + 0.4
    ax.set_ylim(max(0, y_min), y_max)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
# ============ 6. Profile Matching Signal(来自 2b 的巧思)============
st.subheader("🔍 Missingness as a Signal")

match_grp = df.groupby('is_profile_matched').agg(
    impressions=('clk', 'count'),
    clicks=('clk', 'sum')
).reset_index()
match_grp['ctr'] = match_grp['clicks'] / match_grp['impressions'] * 100
match_grp['label'] = match_grp['is_profile_matched'].map(
    {0: 'Unmatched (profile missing)', 1: 'Matched'}
)

col_l, col_r = st.columns([2, 1])
with col_l:
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(match_grp['label'], match_grp['ctr'], color=['#e74c3c', '#2ecc71'])
    ax.axvline(x=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    for bar, val, imp in zip(bars, match_grp['ctr'], match_grp['impressions']):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}% (n={imp:,})', va='center', fontsize=10)
    ax.set_xlabel('CTR (%)')
    ax.set_title('CTR: Profile-matched vs Unmatched Users')
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

# ============ 关键发现总结 ============
st.markdown("---")
st.subheader("🎯 Key Findings & Modeling Recommendations")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **🔝 High-priority features**
    - `age_level` (0.66pp spread)
    - `final_gender_code` (0.39pp)
    - `pvalue_level` (0.37pp)
    - `price_bucket` (monotonic trend)
    - `is_profile_matched` (novel)
    """)

with col_b:
    st.markdown("""
    **⚠️ Limited-signal features**
    - `shopping_level` (0.20pp)
    - `new_user_class_level` (0.20pp)
    - `hour_of_day` (<0.5pp variation)
    - Anonymous ID rankings (`cate_id`, `brand`) — interpretable only in aggregate
    """)

st.success("✅ **Recommendation for modeling**: Prioritize the 5 high-signal features above. "
           "Explore `age × pvalue` and `shopping × pvalue` interactions (not shown here — "
           "see Module 2b for interaction heatmaps with 1.55pp spread).")

# ============ 7. Methodology Note ============
with st.expander("📋 Methodology Note: Missing Value Handling"):
    st.markdown("""
    Different analyses use different missing-value strategies based on their goal:

    - **Single-dimension CTR views**: Retain NaN as a visible category (BigQuery default)
    - **Spread ranking & interaction heatmaps**: `dropna` + minimum sample threshold (n≥500)
    - **Statistical tests**: `dropna` required for chi-square
    - **Clustering (K-means)**: Median imputation (distance-based method cannot handle NaN)
    - **LightGBM**: Native NaN handling
    - **Novel**: `is_profile_matched` feature captures missingness as signal

    **Price outliers**: Capped at p99 (≈5,900 CNY) to reduce skew. Top 1% represents
    luxury segment and does not materially affect CTR patterns.
    """)

st.caption("📊 Data: Alibaba Taobao Advertising Dataset | Dashboard by Wanman Gao | "
           "UCLA MDSH Capstone 2026")