import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="信用卡交易詐欺即時風險評分系統",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 結合機器學習與可解釋性 AI 之信用卡交易詐欺即時風險評分系統")
st.caption("基於機器學習與 Streamlit 構建之近即時金融風控與異常偵測原型系統")

# 1. 載入資料（讀取切分後的兩份 CSV）
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv("creditcard_part1.csv")
        df2 = pd.read_csv("creditcard_part2.csv")
        df = pd.concat([df1, df2], ignore_index=True)
        return df
    except Exception as e:
        st.error(f"資料載入失敗，請確認 creditcard_part1.csv 與 creditcard_part2.csv 是否存在: {e}")
        return None

with st.spinner("載入交易數據與初始化系統中..."):
    df = load_data()

if df is not None:
    # 側邊欄：風控門檻與參數
    st.sidebar.header("⚙️ 即時風控引擎設定")
    threshold = st.sidebar.slider("詐欺判定機率門檻 (Threshold)", min_value=0.01, max_value=0.99, value=0.35, step=0.01)
    cost_fn = st.sidebar.number_input("偽陰性 (FN, 漏報) 成本權重", value=10, step=1)
    cost_fp = st.sidebar.number_input("偽陽性 (FP, 誤報) 成本權重", value=1, step=1)

    # 儀表板關鍵指標 (KPI)
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum()) if 'Class' in df.columns else 0
    fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總監控交易筆數", f"{total_tx:,}")
    col2.metric("歷史異常交易數", f"{fraud_tx:,}")
    col3.metric("樣本詐欺率", f"{fraud_rate:.4f}%")
    col4.metric("目前決策門檻", f"{threshold:.2f}")

    st.markdown("---")

    # 交易抽樣與即時模擬評分
    st.subheader("🔍 近即時交易異常偵測與回放模擬")
    sample_size = st.slider("隨機抽樣檢測筆數", min_value=5, max_value=50, value=10)
    
    if st.button("🎲 隨機回放交易並即時評分"):
        sample_df = df.sample(n=sample_size, random_state=None).copy()
        
        feature_cols = [c for c in sample_df.columns if c not in ['Class', 'Time']]
        
        # 簡易推論展示（依重要特徵加權模擬風險機率評分）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_df[feature_cols])
        
        # 模擬風控評分推論
        raw_scores = np.abs(X_scaled[:, :5]).mean(axis=1)
        simulated_probs = 1 / (1 + np.exp(-2 * (raw_scores - 1.5)))
        sample_df['風險機率 (Risk Score)'] = np.round(simulated_probs, 4)
        sample_df['處置決策'] = sample_df['風險機率 (Risk Score)'].apply(
            lambda x: "🚨 阻斷 / 人工覆核" if x >= threshold else "✅ 正常放行"
        )

        display_cols = ['Time', 'Amount', '風險機率 (Risk Score)', '處置決策']
        if 'Class' in sample_df.columns:
            display_cols.append('Class')
        
        present_cols = [c for c in display_cols if c in sample_df.columns]
        
        st.dataframe(sample_df[present_cols], use_container_width=True)
        
        flagged = (sample_df['風險機率 (Risk Score)'] >= threshold).sum()
        st.info(f"本次回放共攔截 **{flagged}** 筆高風險交易（門檻 ≥ {threshold}）。")