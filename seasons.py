import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 계산기", page_icon="🌿", layout="wide")

@st.cache_data(ttl=3600)
def fetch_data():
    # 데이터 수집 (충분한 계산을 위해 3개월치로 확장)
    qqq = yf.download("QQQ", period="3mo", progress=False)
    soxl = yf.download("SOXL", period="3mo", progress=False)
    
    if len(qqq) < 15 or len(soxl) < 2:
        return None, None, None

    # --- QQQ RSI(14) 계산 ---
    close_qqq = qqq['Close']
    delta = close_qqq.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # 0으로 나누기 방지
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    
    # NaN 값 제외하고 가장 최근 유효 데이터 추출
    last_rsi = float(rsi_series.dropna().iloc[-1])
    
    # SOXL 종가 데이터 (가장 최근 확정된 2일치)
    soxl_closes = soxl['Close'].dropna()
    soxl_p1 = float(soxl_closes.iloc[-1]) # 어제 종가
    soxl_p2 = float(soxl_closes.iloc[-2]) # 그저께 종가
    
    return last_rsi, soxl_p1, soxl_p2

# --- UI 구성 ---
st.title("🌿 사계절 전략(Ivy-Willow-Lily) 대시보드")
st.markdown(f"**최종 업데이트:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    with st.spinner('실시간 시장 데이터를 분석 중입니다...'):
        rsi, p1, p2 = fetch_data()

    if rsi is None:
        st.error("데이터를 충분히 불러오지 못했습니다. 시장 데이터가 아직 업데이트 중일 수 있습니다.")
    else:
        # --- 로직 계산 ---
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        # 친구가 준 RSI 기준 적용 (필요시 65/30 등으로 조정 가능)
        if rsi > 65:
            mode, color, desc = "Ivy (강세)", "red", "상승 에너지가 강합니다. 공격적 매도 목표."
            buy_p = x - 0.01
            sell_p = np.ceil((x * 1.03) * 100) / 100
        elif rsi > 45:
            mode, color, desc = "Willow (정상)", "orange", "안정적인 흐름입니다. 평단가 근처 매매."
            buy_p = x - 0.01
            sell_p = x
        else:
            mode, color, desc = "Lily (하락/조정)", "blue", "하락 압력이 높습니다. 보수적 매수 목표."
            buy_p = np.floor((x * 0.975) * 100) / 100
            sell_p = x

        # --- 결과 표시 ---
        st.divider()
        st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")
        st.info(desc)

        c1, c2, c3 = st.columns(3)
        c1.metric("QQQ RSI (14일)", f"{rsi:.2f}")
        c2.metric("SOXL 어제 종가 (p1)", f"${p1:.2f}")
        c3.metric("SOXL 그저께 종가 (p2)", f"${p2:.2f}")

        st.divider()
        st.subheader("🎯 오늘의 주문 가이드 (LOC)")
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            st.success(f"**매수 주문가:** `${buy_p:.2f}` 이하")
        with col_sell:
            st.error(f"**매도 주문가:** `${sell_p:.2f}` 이상")

except Exception as e:
    st.error(f"데이터 처리 중 오류 발생: {e}")
    st.warning("미국 시장 휴장일이거나 데이터 업데이트 시간(오전 9시~10시 사이)일 수 있습니다.")
