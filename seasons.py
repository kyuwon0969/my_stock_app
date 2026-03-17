import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 대시보드", page_icon="🌿", layout="wide")

@st.cache_data(ttl=300)
def fetch_data():
    # 데이터 수집 (Wilder RSI 계산을 위해 6개월치)
    data = yf.download(["SOXL", "QQQ"], period="6mo", progress=False)
    
    if data.empty:
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        soxl = data['Close']['SOXL'].dropna()
        qqq = data['Close']['QQQ'].dropna()
    else:
        soxl = data['SOXL'].dropna()
        qqq = data['QQQ'].dropna()

    today_date = datetime.now().strftime('%Y-%m-%d')
    
    if soxl.index[-1].strftime('%Y-%m-%d') == today_date:
        p_live = float(soxl.iloc[-1])
        p1 = float(soxl.iloc[-2])
        p2 = float(soxl.iloc[-3])
        qqq_for_rsi = qqq.iloc[:-1]
    else:
        p_live = float(soxl.iloc[-1])
        p1 = float(soxl.iloc[-1])
        p2 = float(soxl.iloc[-2])
        qqq_for_rsi = qqq

    # QQQ RSI 14 계산 (Wilder's Smoothing 방식 적용)
    delta = qqq_for_rsi.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    last_rsi = float(rsi_series.iloc[-1])
    
    return last_rsi, p1, p2, p_live

# --- UI 구성 ---
st.title("🌿 사계절 최적화 전략 (Ivy-Willow-Lily-Tulip)")

# 사이드바: 내 계좌 설정
st.sidebar.header("💰 내 계좌 설정")
init_seed = st.sidebar.number_input("초기 시드 (USD)", min_value=0.0, value=10000.0, step=1000.0)
current_profit = st.sidebar.number_input("누적 수익금 (USD)", min_value=0.0, value=0.0, step=100.0)
total_capital = init_seed + current_profit

st.sidebar.divider()
st.sidebar.write(f"**현재 총 운용 자산:** `${total_capital:,.2f}`")
st.sidebar.write(f"**누적 수익률:** `{(current_profit/init_seed*100) if init_seed > 0 else 0:.2f}%` 기초")

st.markdown(f"**조회 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

try:
    with st.spinner('최적화된 타점을 계산 중입니다...'):
        result = fetch_data()

    if result is None:
        st.error("데이터를 불러오지 못했습니다.")
    else:
        rsi, p1, p2, live = result
        
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi > 65:
            mode, color = "Ivy (강세)", "red"
            buy_limit, sell_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi > 45:
            mode, color = "Willow (정상)", "orange"
            buy_limit, sell_limit = willow_x - 0.01, willow_x
        elif rsi > 30:
            mode, color = "Lily (하락)", "blue"
            buy_limit, sell_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else:
            mode, color = "Tulip (과매도)", "purple"
            buy_limit, sell_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        # --- 상단 지표 ---
        st.divider()
        st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QQQ RSI", f"{rsi:.2f}")
        c2.metric("어제 종가 (p1)", f"${p1:.2f}")
        c3.metric("그저께 종가 (p2)", f"${p2:.2f}")
        c4.metric("SOXL 현재가", f"${live:.2f}", delta=f"{live-p1:.2f}")

        # --- 주문 가이드 및 수량 계산 ---
        st.divider()
        st.subheader("🎯 오늘의 최적화 주문 가이드")
        
        # 5분할 매수 수량 계산
        one_slot_cash = total_capital / 5
        buy_qty = int(one_slot_cash // buy_limit) if buy_limit > 0 else 0
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.success("### 📥 매수 (Buy LOC)")
            st.write(f"**가격:** `${buy_limit:.2f}` 이하")
            st.write(f"**수량:** `{buy_qty}주` 권장")
            st.caption(f"계산 근거: 1슬롯(${one_slot_cash:,.2f}) 기준")
            
        with col_sell:
            st.error("### 📤 매도 (Sell LOC)")
            st.write(f"**가격:** `${sell_limit:.2f}` 이상")
            st.write(f"**수량:** `보유 전량` 매도")
            st.caption(f"목표 수익률: {((sell_limit/willow_x)-1)*100:+.1f}% (x값 대비)")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
