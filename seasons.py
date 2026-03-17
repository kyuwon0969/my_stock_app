import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 계산기", page_icon="🌿", layout="wide")

@st.cache_data(ttl=3600)
def fetch_data():
    # 데이터 수집 (안전하게 2개월치)
    qqq = yf.download("QQQ", period="2mo", progress=False)
    soxl = yf.download("SOXL", period="2mo", progress=False)
    
    # --- QQQ RSI(14) 계산 ---
    delta = qqq['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    
    # 어제(마지막 확정일) 데이터 추출
    last_rsi = float(rsi_series.iloc[-1])
    soxl_p1 = float(soxl['Close'].iloc[-1]) # 어제 종가
    soxl_p2 = float(soxl['Close'].iloc[-2]) # 그저께 종가
    
    return last_rsi, soxl_p1, soxl_p2

# --- UI 구성 ---
st.title("🌿 사계절 전략(Ivy-Willow-Lily) 대시보드")
st.markdown(f"**최종 업데이트:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    with st.spinner('실시간 시장 데이터를 분석 중입니다...'):
        rsi, p1, p2 = fetch_data()

    # --- 로직 계산 ---
    # x = (p1 + p2) * 1.01 / 1.99 (소수점 둘째자리 올림)
    x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
    
    if rsi > 65:
        mode, color = "Ivy (강세)", "red"
        buy_p = x - 0.01
        sell_p = np.ceil((x * 1.03) * 100) / 100
    elif rsi > 45:
        mode, color = "Willow (정상)", "orange"
        buy_p = x - 0.01
        sell_p = x
    else:
        mode, color = "Lily (하락/조정)", "blue"
        buy_p = np.floor((x * 0.975) * 100) / 100
        sell_p = x

    # --- 상단 메트릭 배치 ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("QQQ RSI (14일)", f"{rsi:.2f}")
    c2.metric("SOXL 어제 종가 (p1)", f"${p1:.2f}")
    c3.metric("SOXL 그저께 종가 (p2)", f"${p2:.2f}")

    st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")

    # --- 주문 가격 안내 ---
    st.divider()
    st.subheader("🎯 오늘의 주문 가이드 (LOC)")
    col_buy, col_sell = st.columns(2)
    
    with col_buy:
        st.success(f"**매수 주문가:** `${buy_p:.2f}` 이하")
        st.info(f"계산 근거: { 'x-0.01' if rsi > 45 else 'x * 0.975 (내림)' }")

    with col_sell:
        st.error(f"**매도 주문가:** `${sell_p:.2f}` 이상")
        st.info(f"계산 근거: { 'x * 1.03 (올림)' if rsi > 65 else 'x' }")

    # --- 개인 자산 관리 섹션 ---
    st.divider()
    st.subheader("💰 내 계좌 상태 (입력)")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        my_avg = st.number_input("내 평단가 ($)", value=0.0, step=0.1)
    with ac2:
        my_qty = st.number_input("보유 수량 (주)", value=0, step=1)
    with ac3:
        initial_cash = st.number_input("초기 투자금 ($)", value=1000.0, step=100.0)

    if my_qty > 0:
        current_val = my_qty * p1
        profit = current_val - (my_avg * my_qty)
        return_rate = (current_val / (my_avg * my_qty) - 1) * 100
        
        st.write("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("현재 자산 가치", f"${current_val:,.2f}")
        res2.metric("누적 수익", f"${profit:,.2f}", delta=f"{return_rate:.2f}%")
        res3.metric("총 수익률", f"{ (current_val/initial_cash - 1)*100 :.2f}%")

except Exception as e:
    st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")