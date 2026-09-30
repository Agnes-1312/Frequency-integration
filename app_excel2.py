import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. 頁面配置
st.set_page_config(page_title="FOG 光纖陀螺儀振動診斷系統", layout="wide")

st.title("🔬 光纖陀螺儀 (FOG) 振動監測與 ISO 10816 自動診斷系統")
st.write("本系統結合 快速傅立葉變換 (FFT) 時頻分析技術與 ISO 10816 國際標準 進行機械健康度評估。")

# 2. 側邊欄設定 (全部採用可拖曳的滑桿 Slider)
st.sidebar.header("⚙️ 系統參數設定")

machine_class = st.sidebar.selectbox(
    "選擇 ISO 10816 機械類別：",
    [
        "Class I: 小型馬達 (<15kW)",
        "Class II: 中型馬達 (15~75kW)",
        "Class III: 大型馬達 (剛性基座)",
        "Class IV: 大型馬達 (柔性基座)"
    ]
)

# 採樣頻率拉桿
sampling_rate = st.sidebar.slider(
    "採樣頻率 Sampling Rate (Hz)",
    min_value=1000,
    max_value=20000,
    value=10000,
    step=500
)

st.sidebar.divider()
st.sidebar.header("📁 數據檔案載入")
uploaded_file = st.sidebar.file_uploader("上傳實驗數據 (.xlsx, .csv, .txt)", type=["xlsx", "csv", "txt"])

# 模擬訊號控制拉桿 (未上傳檔案時生效)
st.sidebar.divider()
st.sidebar.header("🌊 光纖訊號模擬設定")
amp_2k5 = st.sidebar.slider("2.5 kHz 特徵頻率振幅 (mm/s)", 0.0, 1.0, 0.20, 0.05)
amp_7k5 = st.sidebar.slider("7.5 kHz 特徵頻率振幅 (mm/s)", 0.0, 1.0, 0.40, 0.05)
noise_level = st.sidebar.slider("背景雜訊強度", 0.0, 0.5, 0.20, 0.05)


# 3. 數據讀取與處理邏輯
time_signal = None
signal_name = ""

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        st.sidebar.success(f"成功讀取檔案：{uploaded_file.name}")
        selected_col = st.sidebar.selectbox("請選擇要分析的數據欄位", df.columns)
        
        raw_data = df[selected_col].dropna().values
        time_signal = raw_data.astype(float)
        signal_name = f"實測數據：{selected_col}"
    except Exception as e:
        st.sidebar.error(f"檔案讀取失敗：{e}")

# 若無上傳檔案，使用側邊欄拉桿調控的模擬數據
if time_signal is None:
    t = np.linspace(0, 0.05, int(sampling_rate * 0.05), endpoint=False) # 取 50ms 顯示
    # 生成包含 2.5kHz 與 7.5kHz 的訊號
    time_signal = amp_2k5 * np.sin(2 * np.pi * 2500 * t) + amp_7k5 * np.sin(2 * np.pi * 7500 * t) + np.random.normal(0, noise_level, len(t))
    signal_name = "示範模擬訊號 (2.5kHz + 7.5kHz)"

# 計算時間軸 (ms)
t_ms = (np.arange(len(time_signal)) / sampling_rate) * 1000

# 【傅立葉變換 FFT 計算】
fft_vals = np.abs(np.fft.rfft(time_signal)) * (2.0 / len(time_signal))
fft_freqs_khz = np.fft.rfftfreq(len(time_signal), 1.0 / sampling_rate) / 1000.0  # 轉成 kHz

# 【RMS 速度有效值計算】
v_rms = np.sqrt(np.mean(time_signal**2))

# 【ISO 10816 門檻值】
if "Class I" in machine_class:
    limits = [0.71, 1.8, 4.5]
elif "Class II" in machine_class:
    limits = [1.12, 2.8, 7.1]
elif "Class III" in machine_class:
    limits = [1.8, 4.5, 11.2]
else:
    limits = [2.8, 7.1, 18.0]

# 健康度判定
if v_rms < limits[0]:
    status_text, status_color, status_icon = "Class A (良好 Good)", "green", "🟢"
elif v_rms < limits[1]:
    status_text, status_color, status_icon = "Class B (良好 Satisfactory)", "gold", "🟡"
elif v_rms < limits[2]:
    status_text, status_color, status_icon = "Class C (警告 Alert)", "orange", "🟠"
else:
    status_text, status_color, status_icon = "Class D (危險 Danger)", "red", "🔴"


# 4. 主要分頁頁面 (Tabs)
tab1, tab2 = st.tabs(["📊 傅立葉時頻分析 (FFT)", "🛡️ ISO 10816 健康度診斷"])

# --- 分頁一：時頻分析 ---
with tab1:
    st.subheader("1. 光纖時域波形與傳立葉頻譜分析")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📈 原始光纖時域訊號 x(t)")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_ms, y=time_signal, mode='lines', name='波形', line=dict(color='#0088FF', width=1)))
        fig_time.update_layout(
            xaxis_title="時間 (ms)",
            yaxis_title="振幅 (mV)",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
    with col_b:
        st.markdown("#### 🌀 快速傳立葉變換頻譜 X(f)")
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=fft_freqs_khz, y=fft_vals, mode='lines', name='頻譜', line=dict(color='#FF5500', width=1.5)))
        fig_fft.update_layout(
            xaxis_title="頻率 (kHz)",
            yaxis_title="頻譜強度 (mm/s)",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_fft, use_container_width=True)

# --- 分頁二：ISO 健康度診斷 ---
with tab2:
    st.subheader("2. ISO 10816 振動健康度自動評估")
    
    # 頂部三張數據指標卡片
    m1, m2, m3 = st.columns(3)
    m1.metric("即時計算速度 RMS 值", f"{v_rms:.2f} mm/s")
    m2.metric("選定 ISO 機械等級", machine_class.split(":")[0])
    m3.markdown(f"### 診斷結果：{status_icon} :{status_color}[{status_text}]")
    
    st.divider()
    
    # 完整 ISO 10816-3 標準門檻對照表
    st.subheader("📋 ISO 10816-3 振動標準判斷門檻表")
    
    iso_data = {
        "評價等級 / 狀態": [
            "0  Class A (優良)",
            "1  Class B (良好)",
            "2  Class C (警告)",
            "3  Class D (危險)"
        ],
        "Class I (小型馬達)": ["< 0.71 mm/s", "0.71 ~ 1.8 mm/s", "1.8 ~ 4.5 mm/s", "> 4.5 mm/s"],
        "Class II (中型馬達)": ["< 1.12 mm/s", "1.12 ~ 2.8 mm/s", "2.8 ~ 7.1 mm/s", "> 7.1 mm/s"],
        "Class III (大型剛性)": ["< 1.8 mm/s", "1.8 ~ 4.5 mm/s", "4.5 ~ 11.2 mm/s", "> 11.2 mm/s"],
        "Class IV (大型柔性)": ["< 2.8 mm/s", "2.8 ~ 7.1 mm/s", "7.1 ~ 18.0 mm/s", "> 18.0 mm/s"]
    }
    
    iso_df = pd.DataFrame(iso_data)
    st.table(iso_df)