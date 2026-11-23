import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="人工マクロポア自律メンテナンス v3.1", page_icon="🌱", layout="wide")
st.title("🌱 人工マクロポア自律メンテナンスシステム")
st.caption("気象予報・土壌状態・蓄電量を統合した予測制御モニター｜v3.1（地名検索・リスクモデル洗練版）")

@st.cache_data(ttl=3600)
def search_locations(query):
    if not query or len(query.strip()) < 2:
        return []
    params = {
        "name": query.strip(),
        "count": 10,
        "language": "ja",
        "format": "json"
    }
    response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=15)
    response.raise_for_status()
    return response.json().get("results", [])

@st.cache_data(ttl=900)
def fetch_weather(lat, lon, timezone):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation,rain,precipitation_probability",
        "forecast_days": 2, "timezone": timezone
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    response.raise_for_status()
    data = pd.DataFrame(response.json()["hourly"])
    data["time"] = pd.to_datetime(data["time"])
    return data

def read_sensor_csv(uploaded):
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    else:
        data = pd.read_csv(BASE / "sensor_data_sample.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in ["soil_moisture", "infiltration_rate_ml_min", "battery_percent", "battery_wh"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    if data.empty:
        raise ValueError("有効なセンサーデータがありません")
    return data

def generate_simulated_sensors(init_inf, curr_inf, moisture, battery_pct):
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(5, -1, -1)]
    infs = [init_inf - (init_inf - curr_inf) * (i / 5.0) for i in range(6)]
    moistures = [max(0.05, moisture - 0.02 * (5 - i)) for i in range(6)]
    batteries = [min(100.0, battery_pct + 0.5 * (5 - i)) for i in range(6)]
    battery_whs = [round(b * 0.092, 2) for b in batteries]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "soil_moisture": moistures,
        "infiltration_rate_ml_min": infs,
        "battery_percent": batteries,
        "battery_wh": battery_whs
    })

def calculate(weather, sensors, refs, thresholds, weights, init_infiltration_ref, tz_str):
    try:
        now = pd.Timestamp.now(tz=tz_str).tz_localize(None)
    except Exception:
        now = pd.Timestamp.now()
    future = weather[weather["time"] >= now].head(6)
    if future.empty:
        future = weather.tail(6)
    rain6 = float(future["precipitation"].fillna(0).sum())
    max_rain = float(future["rain"].fillna(0).max())

    latest = sensors.iloc[-1]
    moisture = float(latest["soil_moisture"])

    inf_series = sensors["infiltration_rate_ml_min"].dropna()
    if not inf_series.empty and init_infiltration_ref > 0:
        recent_inf = float(inf_series.tail(3).mean())
        drop = max(0.0, (init_infiltration_ref - recent_inf) / init_infiltration_ref)
    else:
        recent_inf = 0.0
        drop = 0.0

    norm_rain6 = min(rain6 / refs["rain6"], 1.0)
    norm_intensity = min(max_rain / refs["intensity"], 1.0)
    norm_moisture = min(max(moisture / refs["moisture"], 0.0), 1.0)
    norm_drop = min(drop / refs["drop"], 1.0)

    rain_stress = 0.5 * norm_rain6 + 0.5 * norm_intensity
    soil_vulnerability = 0.6 * norm_drop + 0.4 * norm_moisture
    synergy = rain_stress * soil_vulnerability

    parts = {
        "6時間降水量": norm_rain6,
        "最大時間雨量": norm_intensity,
        "土壌水分": norm_moisture,
        "浸透量低下": norm_drop,
        "降雨×土壌相乗効果": synergy
    }

    score = 100.0 * sum(parts[k] * weights[k] for k in parts)
    score = min(max(score, 0.0), 100.0)

    level = "高" if score >= thresholds[1] else "中" if score >= thresholds[0] else "低"
    battery = float(latest["battery_percent"])
    battery_wh = float(latest["battery_wh"])

    if level == "低":
        action, duration, cycles, reason = "待機", 0.0, 0, "低リスクのため蓄電を優先"
    elif battery < 40 or battery_wh < 1.0:
        action, duration, cycles, reason = "待機", 0.0, 0, "蓄電量不足のため作動を保留"
    elif level == "中":
        action, duration, cycles, reason = "短時間作動案", 0.5, 1, "中リスクに対する試験的な予防作動"
    else:
        action, duration, cycles, reason = "重点作動案", 2.0, 2, "高リスク（降雨×目詰まり進行）に対する上限付き作動案"

    return {
        "score": score, "level": level, "rain6": rain6, "max_rain": max_rain,
        "moisture": moisture, "recent_inf": recent_inf, "drop": drop,
        "battery": battery, "battery_wh": battery_wh,
        "parts": parts, "action": action, "duration": duration,
        "cycles": cycles, "reason": reason
    }

with st.sidebar:
    st.header("観測地点の設定")
    loc_mode = st.radio(
        "地点の指定方法",
        ["地名で検索", "プリセット・直接入力"],
        horizontal=True
    )

    default_lat = float(CFG["location"]["latitude"])
    default_lon = float(CFG["location"]["longitude"])
    default_tz = CFG["location"]["timezone"]
    selected_place_label = f"基準地点 ({default_lat:.4f}, {default_lon:.4f})"

    if loc_mode == "地名で検索":
        place_query = st.text_input("市町村・地名を入力", value="岡山市", placeholder="例: 岡山市, 津島")
        results = []
        if place_query.strip():
            try:
                results = search_locations(place_query)
            except Exception:
                st.warning("地名検索APIに接続できませんでした。直接入力をお試しください。")

        if results:
            options = {}
            for r in results:
                admin = r.get("admin1", "")
                country = r.get("country", "")
                label = f"{r['name']}（{admin} / {country}）"
                options[label] = r
            chosen_label = st.selectbox("候補から選択", list(options.keys()))
            chosen = options[chosen_label]
            lat = float(chosen["latitude"])
            lon = float(chosen["longitude"])
            timezone = chosen.get("timezone", default_tz)
            selected_place_label = chosen_label
        else:
            st.caption("候補が見つかりません。別の地名を入れるか直接入力に切り替えてください。")
            lat, lon, timezone = default_lat, default_lon, default_tz
    else:
        presets = {
            "設定ファイルの基準地点": (default_lat, default_lon, default_tz),
            "岡山市北区（岡山大学周辺）": (34.6889, 133.9222, "Asia/Tokyo"),
            "東京都千代田区": (35.6895, 139.6917, "Asia/Tokyo"),
            "高知県高知市（多雨テスト用）": (33.5597, 133.5311, "Asia/Tokyo"),
            "シンガポール（熱帯スコールテスト用）": (1.3521, 103.8198, "Asia/Singapore"),
        }
        preset_name = st.selectbox("よく使う地点プリセット", list(presets.keys()))
        p_lat, p_lon, p_tz = presets[preset_name]
        lat = st.number_input("緯度", value=p_lat, format="%.4f")
        lon = st.number_input("経度", value=p_lon, format="%.4f")
        timezone = st.text_input("タイムゾーン", value=p_tz)
        selected_place_label = preset_name

    st.caption(f"選択中: {selected_place_label} [{lat:.4f}, {lon:.4f}]")

    st.divider()
    st.subheader("土壌センサーデータ入力")
    data_mode = st.radio(
        "データソース選択",
        ["手動シミュレーション（擬似センサー）", "CSVファイル（サンプル/実測）"],
        index=0
    )

    init_inf_ref = st.number_input("初期健全時の浸透量 (mL/min)", min_value=1.0, value=50.0, step=1.0)

    if data_mode == "手動シミュレーション（擬似センサー）":
        sim_inf = st.slider("現在の浸透量 (mL/min)", min_value=0.0, max_value=60.0, value=38.0, step=0.5)
        sim_moisture = st.slider("現在の土壌水分 (m³/m³)", min_value=0.05, max_value=0.60, value=0.35, step=0.01)
        sim_battery = st.slider("蓄電残量 (%)", min_value=0, max_value=100, value=88, step=1)
        upload = None
    else:
        upload = st.file_uploader("CSVをアップロード", type=["csv"])
        st.caption("未指定の場合は同梱のサンプルCSVを使用")

    st.divider()
    st.subheader("リスク判定の基準値")
    rain_ref = st.number_input("6時間降水量基準 (mm)", min_value=1.0, value=30.0)
    intensity_ref = st.number_input("最大時間雨量基準 (mm/h)", min_value=0.1, value=10.0)
    moisture_ref = st.number_input("土壌水分基準 (m³/m³)", min_value=0.01, max_value=1.0, value=0.40)
    drop_ref = st.number_input("浸透量低下率基準", min_value=0.01, max_value=1.0, value=0.30)
    medium = st.slider("中リスク閾値", 0, 100, 35)
    high = st.slider("高リスク閾値", 1, 100, 65)

try:
    weather = fetch_weather(lat, lon, timezone)
    if data_mode == "手動シミュレーション（擬似センサー）":
        sensors = generate_simulated_sensors(init_inf_ref, sim_inf, sim_moisture, sim_battery)
    else:
        sensors = read_sensor_csv(upload)

    refs = {"rain6": rain_ref, "intensity": intensity_ref, "moisture": moisture_ref, "drop": drop_ref}
    weights = {
        "6時間降水量": 0.20,
        "最大時間雨量": 0.20,
        "土壌水分": 0.15,
        "浸透量低下": 0.25,
        "降雨×土壌相乗効果": 0.20
    }
    result = calculate(weather, sensors, refs, (medium, high), weights, init_inf_ref, timezone)
except Exception as e:
    st.error(f"データ取得または読み込みに失敗しました: {e}")
    st.info("ネット接続を確認するか、CSVの列名・形式を確認してください。")
    st.stop()

st.caption(f"観測地点：{selected_place_label} ｜ 最終更新：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} ｜ 気象データ：Open-Meteo")

c1, c2, c3, c4 = st.columns(4)
c1.metric("目詰まりリスク指標", f"{result['score']:.1f} / 100", result["level"] + "リスク")
c2.metric("今後6時間の降水量", f"{result['rain6']:.1f} mm", f"最大 {result['max_rain']:.1f} mm/h")
c3.metric("土壌水分 / 浸透量", f"{result['moisture']:.3f} m³/m³", f"浸透 {result['recent_inf']:.1f} mL/min (低下 {result['drop']*100:.0f}%)")
c4.metric("蓄電残量", f"{result['battery']:.0f} %", f"{result['battery_wh']:.1f} Wh")

st.progress(min(int(result["score"]), 100), text=f"統合リスクスコア：{result['score']:.1f}")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("降雨予測（今後24時間）")
    try:
        now_local = pd.Timestamp.now(tz=timezone).tz_localize(None)
    except Exception:
        now_local = pd.Timestamp.now()
    chart = weather[weather["time"] >= now_local].head(24).copy()
    if chart.empty:
        chart = weather.head(24).copy()
    y_max = max(5.0, float(chart["precipitation"].fillna(0).max()) * 1.25)
    fig = px.bar(
        chart, x="time", y="precipitation",
        labels={"time": "時刻", "precipitation": "降水量 (mm/h)"},
        range_y=[0, y_max]
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("リスク構成（相乗効果含む）")
    comp = pd.DataFrame({
        "評価項目": list(result["parts"].keys()),
        "正規化値": [v * 100 for v in result["parts"].values()]
    })
    fig2 = px.bar(
        comp, x="正規化値", y="評価項目", orientation="h",
        range_x=[0, 100], labels={"正規化値": "基準に対する割合 (%)"}
    )
    fig2.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("自動制御の判断")
a, b, c = st.columns(3)
a.metric("判断", result["action"])
b.metric("振動時間（案）", f"{result['duration']:.1f} 秒")
c.metric("作動回数（案）", f"{result['cycles']} 回")
st.info(result["reason"])
st.warning("安全上、この画面は制御判断を表示するだけで、圧電素子を駆動しません。実機駆動には素子仕様に適合した専用ドライバ、独立した電流・電圧制限、非常停止、マイコン側の制限機構が必要です。")

st.subheader("土壌センサー履歴")
hist = sensors[["timestamp", "soil_moisture", "infiltration_rate_ml_min"]].melt(
    id_vars="timestamp", var_name="測定項目", value_name="測定値")
fig3 = px.line(hist, x="timestamp", y="測定値", color="測定項目", markers=True)
fig3.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig3, use_container_width=True)
st.dataframe(sensors.tail(10), use_container_width=True, hide_index=True)

st.caption("研究上の注意：このスコアは目詰まり確率ではなく、降雨負荷と土壌状態の相乗効果を考慮した仮のルールベース指標です。")