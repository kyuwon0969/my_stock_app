import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from datetime import datetime
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 매니저 V3", page_icon="🌿", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 ---
@st.cache_data(ttl=600)
def get_processed_data(ticker, start_date):
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        data = yf.download([ticker, "QQQ"], start=fetch_start, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
        else:
            target_close, qqq_close = data['Close'].ffill(), data['Close'].ffill()
            
        df = pd.DataFrame(index=target_close.index)
        df['close'], df['qqq_close'] = target_close, qqq_close
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 (정량 매수 적용) ---
def run_simulation(df, initial_seed, num_slots):
    """정량 매수 로직: 주문 시점(타점) 가격으로 수량을 고정하고 종가에 체결"""
    if df is None or df.empty: return pd.DataFrame(), []
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve = 0.0  
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    history, slot_details = [], []
    qqq_start_p = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
            
        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        # 모드별 타점 설정
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        sold = False
        if shares > 0 and curr_c >= s_l:
            profit = (shares * curr_c) - (avg_price * shares)
            if mode == "Ivy" and profit > 0:
                cash += (avg_price * shares); ivy_reserve += profit        
            else: cash += (shares * curr_c)
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        # [핵심] 정량 매수 로직 (타점 b_l 기준으로 수량 확정)
        if not sold and used_slots < num_slots and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            
            # 수량은 주문을 걸었던 타점(b_l) 기준으로 미리 계산 (정량)
            buy_qty = slot_cash // b_l 
            # 실제 지불은 유리한 종가(curr_c)로 정산
            actual_cost = buy_qty * curr_c
            
            if buy_qty > 0 and cash >= actual_cost:
                slot_details.append({
                    "슬롯": len(slot_details) + 1, "날짜": date.strftime('%Y-%m-%d'),
                    "매수가(종가)": round(float(curr_c), 2), "기준가(타점)": round(float(b_l), 2),
                    "수량": int(buy_qty), "금액": round(float(actual_cost), 2)
                })
                avg_price = ((avg_price * shares) + actual_cost) / (shares + buy_qty)
                shares += buy_qty; cash -= actual_cost; used_slots += 1
        
        history.append({
            'Date': date, 'Total': cash + (shares * curr_c) + ivy_reserve,
            'Ivy': ivy_reserve, 'Cash': cash, 'Shares': shares, 'Slots': used_slots, 
            'Avg': avg_price, 'QQQ': (initial_seed / qqq_start_p) * row['qqq_close']
        })
    return pd.DataFrame(history).set_index('Date'), slot_details

# --- 리스크 분석 엔진 ---
def analyze_metrics(df_res, seed):
    rets = df_res['Total'].pct_change().dropna()
    ann_ret = (df_res['Total'].iloc[-1] / seed) ** (252 / len(df_res)) - 1
    
    rf_daily = (1 + 0.053)**(1/252) - 1
    excess = rets - rf_daily
    sharpe = np.sqrt(252) * excess.mean() / rets.std()
    sortino = np.sqrt(252) * excess.mean() / rets[rets < 0].std()
    
    peak = df_res['Total'].cummax()
    dd = (df_res['Total'] - peak) / peak
    mdd = dd.min()
    
    # DSR (비교군 10개 보정)
    T, sk, kr = len(rets), skew(rets), kurtosis(rets)
    sigma_sr = np.sqrt((1 + 0.5 * sk**2 + (kr-3)/4 * sharpe**2) / (T-1))
    dsr = norm.cdf((sharpe - np.sqrt(2 * np.log(10))) / sigma_sr)

    return {"CAGR": ann_ret*100, "Sharpe": sharpe, "Sortino": sortino, "MDD": mdd*100, "DSR": dsr}

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목", ["SOXL", "USD"], index=0)
    config_key = f"v3_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("슬롯", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("원금", value=float(saved['init_seed']), step=1000.0)
    if st.button("💾 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")

tab1, tab2 = st.tabs(["🎯 실전 가이드", "📊 전문 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        res_live, slots_live = run_simulation(df_live, init_seed, num_slots)
        cur, last = res_live.iloc[-1], df_live.iloc[-1]
        
        st.subheader(f"📊 {target_ticker} 현황")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
        m2.metric("평균단가", f"${cur['Avg']:.2f}")
        m3.metric("진행슬롯", f"{int(cur['Slots'])} / {num_slots}")
        m4.metric("총 자산", f"${cur['Total']:,.2f}")

        if slots_live:
            st.markdown("#### 📝 보유 슬롯 상세 (정량 매수 결과)")
            st.table(pd.DataFrame(slots_live))

        st.divider()
        rsi_n, p1, p2 = last['rsi'], last['p1_c'], last['p2_c']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        st.write(f"🔍 **판단근거:** RSI `{rsi_n:.2f}` | p1 `${p1:.2f}` | p2 `${p2:.2f}`")
        
        cl, cr = st.columns(2)
        with cl:
            st.success(f"#### 📥 {int(cur['Slots'])+1}회차 매수")
            if cur['Slots'] < num_slots:
                b_price = x - 0.01 if rsi_n > 45 else np.floor((x * 0.975) * 100) / 100
                t_cash = cur['Cash'] / (num_slots - cur['Slots']) if cur['Slots'] == 0 else (cur['Total']/num_slots) # 유연한 계산
                st.write(f"타점: `${b_price:.2f}` (LOC)")
                st.write(f"수량: `{int(t_cash // b_price)} 주` (정량 고정)")
            else: st.write("✅ 풀매수")
        with cr:
            s_price = np.ceil((x * 1.03) * 100) / 100 if rsi_n > 65 else x
            st.error("#### 📤 전량 매도")
            if cur['Shares'] > 0: st.write(f"가격: `${s_price:.2f}` (LOC) | `{int(cur['Shares'])} 주`")
            else: st.write("보유 없음")
        st.line_chart(res_live[['Total', 'QQQ']])

with tab2:
    if st.button("🚀 13년 전문 백테스트 실행"):
        df_back = get_processed_data(target_ticker, "2013-01-01")
        if df_back is not None:
            res_b, _ = run_simulation(df_back, init_seed, num_slots)
            m = analyze_metrics(res_b, init_seed)
            st.divider()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CAGR", f"{m['CAGR']:.2f}%")
            c2.metric("MDD", f"{m['MDD']:.2f}%")
            c3.metric("Sharpe", f"{m['Sharpe']:.3f}")
            c4.metric("Sortino", f"{m['Sortino']:.3f}")
            c5.metric("DSR", f"{m['DSR']:.4f}")
            st.line_chart(res_b[['Total', 'QQQ']])
            
            res_b['year'] = res_b.index.year
            y_data, t_seed = [], init_seed
            for yr in sorted(res_b['year'].unique()):
                y_df = res_b[res_b['year'] == yr]
                y_v = y_df['Total'].iloc[-1]
                y_m = (y_df['Total'] / y_df['Total'].cummax() - 1).min() * 100
                y_data.append({'연도': yr, '수익률': f"{(y_v/t_seed-1)*100:.2f}%", 'MDD': f"{y_m:.2f}%"})
                t_seed = y_v
            st.table(pd.DataFrame(y_data))
