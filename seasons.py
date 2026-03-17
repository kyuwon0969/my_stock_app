import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 대시보드", page_icon="🌿", layout="wide")

@st.cache_data(ttl=300)
def fetch_data():
    # 데이터 수집 (Wilder RSI와 p2 계산을 위해 6개월치)
    data = yf.download(["SOXL", "QQQ"], period="6mo", progress=False)
    
    if data.empty:
        return None
    
    # 데이터 정리 (멀티인덱스 대응)
    if isinstance(data.columns, pd.MultiIndex):
        soxl = data['Close']['SOXL'].dropna()
        qqq = data['Close']['QQQ'].dropna()
    else:
        soxl = data['SOXL'].dropna()
        qqq = data['QQQ'].dropna()

    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # 장중 데이터 처리 (오늘 데이터가 있으면 제외하고 p1, p2 확정)
    if soxl.index[-1].strftime('%Y-%m-%d') == today_date:
        p_live = float(soxl.iloc[-1])
        p1 = float(soxl.iloc[-2])
        p2 = float(soxl.iloc[-3])
        qqq_for_rsi = qqq.iloc[:-1] # RSI는 확정된 데이터로만
    else:
        p_live = float(soxl.iloc[-1])
        p1 = float(soxl.iloc[-1])
        p2 = float(soxl.iloc[-2])
        qqq_for_rsi = qqq

    # QQQ RSI 14 계산 (백테스트 코드와 동일한 SMA 방식 유지)
    delta = qqq_for_rsi.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    last_rsi = float(rsi_series.iloc[-1])
    
    return last_rsi, p1, p2, p_live

# --- UI 구성 ---
st.title("🌿 사계절 최적화 전략 (Ivy-Willow-Lily-Tulip)")
st.markdown(f"**조회 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

try:
    with st.spinner('최적화된 타점을 계산 중입니다...'):
        result = fetch_data()

    if result is None:
        st.error("데이터를 불러오지 못했습니다.")
    else:
        rsi, p1, p2, live = result
        
        # 1. 기초 x값 계산 (백테스트 로직)
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # 2. 모드 결정 및 타점 적용
        if rsi > 65:
            mode, color = "Ivy (강세)", "red"
            buy_limit = willow_x - 0.01
            sell_limit = np.ceil((willow_x * 1.03) * 100) / 100 # +3.0%
        elif rsi > 45:
            mode, color = "Willow (정상)", "orange"
            buy_limit = willow_x - 0.01
            sell_limit = willow_x # +0.0%
        elif rsi > 30:
            mode, color = "Lily (하락)", "blue"
            buy_limit = np.floor((willow_x * 0.975) * 100) / 100 # -2.5%
            sell_limit = willow_x
        else:
            mode, color = "Tulip (과매도)", "purple"
            buy_limit = np.floor((willow_x * 0.975) * 100) / 100 # -2.5%
            sell_limit = willow_x # +0.0%

        # --- 메트릭 표시 ---
        st.divider()
        st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QQQ RSI", f"{rsi:.2f}")
        c2.metric("어제 종가 (p1)", f"${p1:.2f}")
        c3.metric("그저께 종가 (p2)", f"${p2:.2f}")
        c4.metric("SOXL 현재가", f"${live:.2f}", delta=f"{live-p1:.2f}")

        # --- 주문 가이드 섹션 ---
        st.divider()
        st.subheader("🎯 오늘의 최적화 LOC 주문 가이드")
        
        st.info(f"**기준 x값 (Willow_x):** `${willow_x:.2f}`")
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.success("### 📥 매수 (Buy LOC)")
            st.write(f"**주문 가격:** `${buy_limit:.2f}` 이하")
            st.caption("※ 5분할 매수 전략 중 1회분 진입 권장")
            
        with col_sell:
            st.error("### 📤 매도 (Sell LOC)")
            st.write(f"**주문 가격:** `${sell_limit:.2f}` 이상")
            st.caption(f"※ 전량 매도 (수익 목표: {((sell_limit/willow_x)-1)*100:+.1f}%)")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
