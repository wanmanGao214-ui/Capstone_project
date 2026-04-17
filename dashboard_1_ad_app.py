import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# ============ 页面配置 ============
st.set_page_config(page_title="Alibaba CTR Dashboard", layout="wide")
st.title("🛍️ Alibaba CTR Prediction — Business Overview")
st.caption("Based on full 1M-row sample. Overall CTR baseline = 5.16%")

# ============ 常量:label 映射(来自 2b)============
DIMENSIONS = {
    'final_gender_code': {1: 'Male', 2: 'Female'},
    'age_level':         {1:'<18', 2:'18-24', 3:'25-29', 4:'30-34',
                          5:'35-39', 6:'40-49', 7:'50+'},
    'pvalue_level':      {1: 'Low', 2: 'Mid', 3: 'High'},
    'shopping_level':    {1: 'Shallow', 2: 'Moderate', 3: 'Deep'},
    'new_user_class_level': {1:'Tier-1', 2:'Tier-2', 3:'Tier-3', 4:'Rural/Other'}
}
MIN_IMPRESSIONS = 500  # 来自 2b,小样本过滤阈值
BASELINE_CTR = 5.16    # 来自 notebook 的整体 CTR

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
st.subheader("🎯 CTR Spread by User Dimension")
st.caption("How much does CTR vary across each dimension? Larger spread = stronger modeling signal.")

spread_data = []
for dim, label_map in DIMENSIONS.items():
    sub = df[[dim, 'clk']].dropna()  # 全量数据算 spread,不受 filter 影响
    grp = sub.groupby(dim)['clk'].agg(['count', 'mean']).reset_index()
    grp = grp[grp['count'] >= MIN_IMPRESSIONS]
    if len(grp) > 0:
        ctrs = grp['mean'] * 100
        spread_data.append({
            'dimension': dim,
            'min_ctr': ctrs.min(),
            'max_ctr': ctrs.max(),
            'spread': ctrs.max() - ctrs.min()
        })

spread_df = pd.DataFrame(spread_data).sort_values('spread', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71' if s > 0.3 else '#e67e22' if s > 0.15 else '#95a5a6' 
          for s in spread_df['spread']]
ax.barh(spread_df['dimension'], spread_df['spread'], color=colors)
for i, (dim, spread) in enumerate(zip(spread_df['dimension'], spread_df['spread'])):
    ax.text(spread + 0.01, i, f"{spread:.2f}pp", va='center', fontsize=10)
ax.set_xlabel('CTR Spread (percentage points)')
ax.set_title('User Dimension Impact Ranking')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info("💡 **Modeling Implication**: Dimensions with larger spread (green) deserve higher "
        "priority in feature engineering. Low-spread dimensions add noise without signal.")

st.markdown("---")

# ============ 3. Temporal Trends ============
st.subheader("⏰ Temporal Trends")

col_l, col_r = st.columns(2)

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
           alpha=0.6, label='Impressions')
    ax2.plot(hourly['hour'], hourly['ctr'], color='coral',
             marker='o', linewidth=2, label='CTR')
    ax2.axhline(y=BASELINE_CTR, color='red', linestyle='--',
                alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    ax.set_xlabel('Hour (Beijing time)')
    ax.set_ylabel('Impressions', color='steelblue')
    ax2.set_ylabel('CTR (%)', color='coral')
    ax.set_title('Hourly Pattern')
    ax2.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

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
    colors_day = ['steelblue' if d < 5 else 'coral' for d in daily['day_of_week']]
    ax.bar(daily['day_name'], daily['ctr'], color=colors_day)
    ax.axhline(y=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    ax.set_ylabel('CTR (%)')
    ax.set_title('CTR by Day of Week (blue=weekday, coral=weekend)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

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
    bars = ax.bar(pid_grp['pid'], pid_grp['ctr'], color=['steelblue', 'coral'])
    ax.axhline(y=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    for bar, val in zip(bars, pid_grp['ctr']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}%', ha='center', fontsize=10)
    ax.set_ylabel('CTR (%)')
    ax.set_title('CTR by Ad Placement')
    ax.tick_params(axis='x', rotation=15)
    ax.legend()
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
    ax.bar(price_grp['price_bucket'].astype(str), price_grp['ctr'], color='coral')
    ax.axhline(y=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    ax.set_ylabel('CTR (%)')
    ax.set_xlabel('Price Range (CNY)')
    ax.set_title('CTR by Price Range (p99 capped)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.info("💡 **Finding**: Price shows a clear monotonic decrease — the 0-100 CNY sweet spot is "
        "consistent across user segments. Ad placement difference is modest (~0.3pp).")

st.markdown("---")

# ============ 5. User Segment CTR ============
st.subheader("👥 CTR by User Segment")

seg_col1, seg_col2 = st.columns(2)

def plot_segment(ax, dim, label_map, title):
    grp = df_f[[dim, 'clk']].dropna().groupby(dim).agg(
        impressions=('clk', 'count'),
        clicks=('clk', 'sum')
    ).reset_index()
    grp['ctr'] = grp['clicks'] / grp['impressions'] * 100
    grp['label'] = grp[dim].map(label_map)
    # 过滤掉 map 后仍为 NaN 的行(label_map 里没有的 key)
    grp = grp.dropna(subset=['label'])
    grp['label'] = grp['label'].astype(str)
    
    bars = ax.bar(grp['label'], grp['ctr'], color='steelblue')
    ax.axhline(y=BASELINE_CTR, color='red', linestyle='--',
               alpha=0.5, label=f'Baseline {BASELINE_CTR}%')
    for bar, val in zip(bars, grp['ctr']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}', ha='center', fontsize=9)
    ax.set_title(title)
    ax.set_ylabel('CTR (%)')
    ax.legend(fontsize=8)

with seg_col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_segment(ax, 'age_level', DIMENSIONS['age_level'], 'CTR by Age Group')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with seg_col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_segment(ax, 'pvalue_level', DIMENSIONS['pvalue_level'], 'CTR by Consumption Level')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

seg_col3, seg_col4 = st.columns(2)

with seg_col3:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_segment(ax, 'shopping_level', DIMENSIONS['shopping_level'], 'CTR by Shopping Depth')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with seg_col4:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_segment(ax, 'new_user_class_level', DIMENSIONS['new_user_class_level'], 'CTR by City Tier')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

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
    st.info("💡 **Insight**: Profile-missing rows are not noise — they form a distinct "
            "segment. We engineer `is_profile_matched` as a binary feature to capture "
            "this signal, rather than discarding or imputing the information.")

st.markdown("---")

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