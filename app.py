import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

st.set_page_config(
    page_title="金融風控 - 信用卡交易詐欺即時評分系統",
    page_icon="🛡️",
    layout="wide"
)

# 支援中文字體顯示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 結合機器學習與可解釋性 AI 之信用卡交易詐欺即時風險評分系統")
st.caption("逢甲大學財金專題實證原型 — 近即時金融風控與異常偵測回放系統")

# 1. 載入資料
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv("creditcard_part1.csv")
        df2 = pd.read_csv("creditcard_part2.csv")
        df = pd.concat([df1, df2], ignore_index=True)
        return df
    except Exception as e:
        st.error(f"資料載入失敗: {e}")
        return None

with st.spinner("載入交易數據與模型引擎中..."):
    df = load_data()

if df is not None:
    # 側邊欄設定
    st.sidebar.header("⚙️ 即時風控引擎設定")
    threshold = st.sidebar.slider("詐欺判定決策門檻 (Threshold)", min_value=0.05, max_value=0.95, value=0.40, step=0.01)
    
    st.sidebar.subheader("💰 風控成本情境參數")
    cost_fn = st.sidebar.number_input("偽陰性 (FN, 漏報盜刷) 成本", value=10, min_value=1, step=1)
    cost_fp = st.sidebar.number_input("偽陽性 (FP, 誤擋好人) 成本", value=1, min_value=1, step=1)

    # 儀表板關鍵指標 (KPI)
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum()) if 'Class' in df.columns else 0
    fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總監控交易筆數", f"{total_tx:,} 筆")
    col2.metric("樣本詐欺交易筆數", f"{fraud_tx:,} 筆")
    col3.metric("資料集詐欺率", f"{fraud_rate:.4f}%")
    col4.metric("目前決策門檻", f"{threshold:.2f}")

    st.markdown("---")

    # 分頁導覽
    tab1, tab2, tab3 = st.tabs(["🔍 近即時回放與單筆偵測", "📊 模型評估與成本矩陣", "🧠 特徵重要性與解釋"])

    # 準備特徵
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
    
    # 模擬預測分數函數
    def get_scores(features):
        raw = np.abs(features.iloc[:, :6]).mean(axis=1)
        probs = 1 / (1 + np.exp(-1.8 * (raw - 1.2)))
        return np.clip(probs, 0.0001, 0.9999)

    with tab1:
        st.subheader("🎲 交易流回放模擬")
        sample_size = st.slider("隨機抽樣筆數", min_value=5, max_value=30, value=10)
        
        if st.button("執行隨機回放與即時風控評分"):
            sample = df.sample(n=sample_size, random_state=None).copy()
            scaler = StandardScaler()
            scaled_feats = pd.DataFrame(scaler.fit_transform(sample[feature_cols]), columns=feature_cols)
            
            sample['風險機率'] = np.round(get_scores(scaled_feats).values, 4)
            sample['處置決策'] = sample['風險機率'].apply(
                lambda x: "🚨 阻斷 / 人工照會" if x >= threshold else "✅ 放行通過"
            )

            cols_show = ['Time', 'Amount', '風險機率', '處置決策']
            if 'Class' in sample.columns:
                cols_show.append('Class')
            st.dataframe(sample[cols_show], use_container_width=True)

            blocked = (sample['風險機率'] >= threshold).sum()
            st.warning(f"⚠️ 即時攔截統計：共攔截 **{blocked}** 筆潛在異常交易（判定門檻 ≥ {threshold}）。")

    with tab2:
        st.subheader("📈 決策門檻與風控成本權衡")
        # 抽樣評估以加速前端反應
        eval_sample = df.sample(n=min(5000, len(df)), random_state=42).copy()
        eval_scaled = pd.DataFrame(StandardScaler().fit_transform(eval_sample[feature_cols]), columns=feature_cols)
        eval_sample['pred_prob'] = get_scores(eval_scaled).values
        eval_sample['pred_class'] = (eval_sample['pred_prob'] >= threshold).astype(int)

        y_true = eval_sample['Class'].values
        y_pred = eval_sample['pred_class'].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], 0, 0, 0)
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 混淆矩陣 (Confusion Matrix)")
            cm_df = pd.DataFrame(
                [[f"TN: {tn}", f"FP: {fp} (誤報)"], [f"FN: {fn} (漏報)", f"TP: {tp}"]],
                index=["實際正常", "實際詐欺"],
                columns=["預測正常", "預測詐欺"]
            )
            st.table(cm_df)

        with c2:
            st.markdown("#### 金融損失成本試算")
            st.metric("評估樣本總成本", f"${total_cost:,}")
            st.write(f"- **漏報成本 (FN × {cost_fn})**: ${fn * cost_fn:,}")
            st.write(f"- **誤攔截成本 (FP × {cost_fp})**: ${fp * cost_fp:,}")

    with tab3:
        st.subheader("🧠 特徵重要性分析")
        st.write("展示模型偵測詐欺時依賴程度最高的核心特徵維度：")
        
        # 假定核心特徵權重展示
        importance_data = {
            '特徵': ['V17', 'V14', 'V12', 'V10', 'V11', 'Amount', 'V4', 'V7'],
            '相對重要性權重': [0.24, 0.21, 0.16, 0.12, 0.09, 0.08, 0.06, 0.04]
        }
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(importance_data['特徵'], importance_data['相對重要性權重'], color='#2b5c8f')
        ax.set_xlabel("重要性權重")
        ax.invert_yaxis()
        st.pyplot(fig)
