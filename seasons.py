import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 매니저", page_icon="🌿", layout="wide")

# 로컬 스토리지 초기화
localS = LocalStorage()

@st.cache_data(ttl=300)
def fetch_data():
    # 데이터 수집 (안정성을 위해 6개월치)
    data = yf.download(["SOXL", "QQQ"], period="6mo", progress=False)
    
    if data.empty or len(data) < 20:
        return None
    
    # 멀티인덱스 대응 및 결측치 제거
    if isinstance(data.columns, pd.MultiIndex):
        soxl = data['Close']['SOXL'].dropna()
        qqq = data['Close']['QQQ'].dropna()
    else:
        soxl = data['SOXL'].dropna()
        qqq = data['QQQ'].dropna()

    if len(soxl) < 10 or len(qqq) < 20:
        return None

    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # 장중/종가 데이터 구분 및 p1, p2 확정
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

    # Wilder's RSI 계산 (트레이딩뷰 방식)
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
st.title("🌿 사계절 최적화 전략 매니저")

# 사이드바: 내 계좌 설정 및 브라우저 저장
with st.sidebar:
    st.header("👤 내 계좌 설정")
    user_name = st.text_input("사용자 이름", value="내 이름")
    
    storage_key = f"seasons_data_{user_name}"
    # 로컬 스토리지에서 데이터 로드
    saved_data = localS.getItem(storage_key)
    if not saved_data:
        saved_data = {"seed": 10000.0, "profit": 0.0, "slot": 0}
    
    init_seed = st.number_input("초기 시드 (USD)", value=float(saved_data['seed']), step=1000.0)
    current_profit = st.number_input("누적 수익금 (USD)", value=float(saved_data['profit']), step=100.0)
    current_slot = st.slider("현재 매수 완료 회차", 0, 5, int(saved_data['slot']))
    
    total_capital = init_seed + current_profit
    one_slot_cash = total_capital / 5
    
    if st.button("💾 이 기기에 데이터 저장"):
        new_data = {"seed": init_seed, "profit": current_profit, "slot": current_slot}
        localS.setItem(storage_key, new_data)
        st.success("브라우저에 저장되었습니다!")
    
    st.divider()
    st.write(f"**현재 총 운용 자산:** `${total_capital:,.2f}`")
    st.write(f"**누적 수익률:** `{(current_profit/init_seed*100) if init_seed > 0 else 0:.2f}%` 기초")

st.markdown(f"**조회 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

# 메인 대시보드 로직
try:
    with st.spinner('데이터를 분석 중입니다...'):
        result = fetch_data()

    if result is None:
        st.warning("📡 시장 데이터를 불러오는 중입니다. 잠시 후 새로고침 해주세요.")
        st.info("미국 시장 휴장일이거나 데이터 업데이트 전일 수 있습니다.")
    else:
        rsi, p1, p2, live = result
        
        # Willow_x 계산 및 모드별 타점 적용
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

        # 상단 지표 출력
        st.divider()
        st.markdown(f"### 현재 시장 모드: :{color}[{mode}]")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QQQ RSI", f"{rsi:.2f}")
        c2.metric("어제 종가 (p1)", f"${p1:.2f}")
        c3.metric("그저께 종가 (p2)", f"${p2:.2f}")
        c4.metric("SOXL 현재가", f"${live:.2f}", delta=f"{live-p1:.2f}")

        # 주문 가이드 및 수량 계산
        st.divider()
        st.subheader("🎯 실전 LOC 주문 가이드")
        
        buy_qty = int(one_slot_cash // buy_limit) if buy_limit > 0 else 0
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.success(f"### 📥 {current_slot + 1}회차 매수 시그널")
            if current_slot < 5:
                st.write(f"**가격:** `${buy_limit:.2f}` 이하")
                st.write(f"**수량:** `{buy_qty}주` 권장")
                st.caption(f"기준: 1슬롯 (${one_slot_cash:,.2f})")
            else:
                st.write("✅ **모든 슬롯 체결 완료** (추가 매수 없음)")
            
        with col_sell:
            st.error("### 📤 전량 매도 시그널")
            st.write(f"**가격:** `${sell_limit:.2f}` 이상")
            st.write(f"**목표 수익:** {((sell_limit/willow_x)-1)*100:+.1f}% (x값 대비)")
            st.caption(f"매도 대상: 보유 중인 전량")

except Exception as e:
    st.error(f"데이터 처리 중 오류 발생: {e}")
