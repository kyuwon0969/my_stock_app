import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 계산기", page_icon="🌿", layout="wide")

@st.cache_data(ttl=300) # 데이터 갱신 주기를 5분으로 단축
def fetch_data():
    # 데이터 수집 (충분한 계산을 위해 3개월치)
    qqq = yf.download("QQQ", period="3mo", progress=False)
    soxl = yf.download("SOXL", period="3mo", progress=False)
    
    if len(qqq) < 20 or len(soxl) < 5:
        return None, None, None, None

    # --- 실시간 장중 데이터 처리 로직 ---
    # yfinance는 장중일 때 마지막 행에 실시간 데이터를 넣습니다. 
    # 사계절 전략은 '확정된 전일 종가'가 필요하므로, 마지막 데이터가 오늘 날짜면 제외합니다.
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # SOXL 처리
    if soxl.index[-1].strftime('%Y-%m-%d') == today_date:
        # 오늘 데이터가 포함되어 있다면 그 전 데이터들을 사용
        soxl_p1 = float(soxl['Close'].iloc[-2]) # 어제 확정 종가
        soxl_p2 = float(soxl['Close'].iloc[-3]) # 그저께 확정 종가
        current_price = float(soxl['Close'].iloc[-1]) # 현재 움직이는 가격
    else:
        # 아직 오늘 장이 안 열렸거나 데이터가 안 들어왔다면
        soxl_p1 = float(soxl['Close'].iloc[-1])
        soxl_p2 = float(soxl['Close'].iloc[-2])
        current_price = soxl_p1

    # --- QQQ RSI(14) 계산 ---
    # RSI 계산 시에도 장중 데이터는 제외하고 확정 데이터로만 계산
    qqq_final = qqq.iloc[:-1] if qqq.index[-1].strftime('%Y-%m-%d') == today_date else qqq
    
    close_qqq = qqq_final['Close']
    delta = close_qqq.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    last_rsi = float(rsi_series.dropna().iloc[-1])
    
    return last_rsi, soxl_p1, soxl_p2, current_price

# --- UI 구성 ---
st.title("🌿 사계절 전략(Ivy-Willow-Lily) 대시보드")
st.markdown(f"**조회 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

try:
    with st.spinner('데이터를 정밀 분석 중입니다...'):
        rsi, p1, p2, live = fetch_data()

    if rsi is None:
        st.error("데이터 로드 실패")
    else:
        # --- 로직 계산 ---
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        if rsi > 65:
            mode, color = "Ivy (강세)", "red"
            buy_p, sell_p = x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi > 45:
            mode, color = "Willow (정상)", "orange"
            buy_p, sell_p = x - 0.01, x
        else:
            mode, color = "Lily (하락)", "blue"
            buy_p, sell_p = np.floor((x * 0.975) * 100) / 100, x

        # --- 대시보드 표시 ---
        st.divider()
        st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("QQQ RSI", f"{rsi:.2f}")
        m2.metric("어제 종가 (p1)", f"${p1:.2f}")
        m3.metric("그저께 종가 (p2)", f"${p2:.2f}")
        m4.metric("SOXL 현재가", f"${live:.2f}", delta=f"{live-p1:.2f}")

        st.divider()
        st.subheader("🎯 오늘의 LOC 주문 가이드")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.success(f"**매수 지정가:** `${buy_p:.2f}`")
        with col_sell:
            st.error(f"**매도 지정가:** `${sell_p:.2f}`")
        
        st.caption(f"※ 현재 미국 장중인 경우 'SOXL 현재가'는 실시간이며, 주문가는 어제/그저께 확정 종가 기반입니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
