import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 자동 매니저", page_icon="🌿", layout="wide")

# --- 데이터 수집 함수 ---
@st.cache_data(ttl=3600)
def get_full_data(ticker, start_date):
    """시작일부터 현재까지의 전체 데이터를 가져옴"""
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
    end_date = datetime.now() + timedelta(days=1)
    data = yf.download([ticker, "QQQ"], start=fetch_start, end=end_date.strftime('%Y-%m-%d'), progress=False)
    if data.empty: return None
    
    if isinstance(data.columns, pd.MultiIndex):
        target_close = data['Close'][ticker].dropna()
        qqq_close = data['Close']['QQQ'].dropna()
    else:
        target_close = data[ticker].dropna()
        qqq_close = data['QQQ'].dropna()
    
    df = pd.DataFrame(index=target_close.index)
    df['close'] = target_close
    df['prev_close'] = df['close'].shift(1)
    df['prev_close2'] = df['close'].shift(2)
    
    # QQQ 기반 RSI 계산
    delta = qqq_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
    
    return df.loc[start_date:].dropna()

def calculate_current_status(df, initial_seed):
    """데이터를 돌며 현재 보유 수량, 평단가, 남은 현금, 채워진 슬롯 계산"""
    cash = initial_seed
    shares = 0
    used_slots = 0
    slot_cash = 0
    avg_price = 0
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, rsi_val = row['prev_close'], row['prev_close2'], row['close'], row['rsi']
        
        # x값 및 기준가 계산
        x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # 모드별 매수/매도 제한가
        if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        # 1. 매도 체크
        if shares > 0 and curr_close >= s_limit:
            cash += (shares * curr_close)
            shares, used_slots, slot_cash, avg_price = 0, 0, 0, 0
        
        # 2. 매수 체크
        if used_slots < 5 and curr_close <= b_limit:
            if used_slots == 0: 
                slot_cash = cash / 5
            if cash >= slot_cash:
                buy_qty = slot_cash // curr_close
                if buy_qty > 0:
                    new_total_cost = (avg_price * shares) + (buy_qty * curr_close)
                    shares += buy_qty
                    avg_price = new_total_cost / shares
                    cash -= (buy_qty * curr_close)
                    used_slots += 1
                    
    return cash, shares, avg_price, used_slots

# --- UI 레이아웃 ---
st.title("🌿 사계절 전략 자동 추적 매니저")

with st.sidebar:
    st.header("⚙️ 투자 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD", "QLD"], index=0)
    
    st.divider()
    # 시작일을 입력받음
    start_date = st.date_input("운용 시작일", value=datetime(2024, 1, 1))
    init_seed = st.number_input("투자 원금 (USD)", value=10000.0, step=1000.0)
    
    st.divider()
    st.caption("※ 시작일부터 현재까지의 매매 기록을 자동으로 시뮬레이션하여 현재 상태를 계산합니다.")

# 데이터 로드 및 현재 상태 계산
df_all = get_full_data(target_ticker, start_date.strftime('%Y-%m-%d'))

if df_all is not None and not df_all.empty:
    current_cash, current_shares, current_avg, current_slots = calculate_current_status(df_all, init_seed)
    
    # 마지막 거래일 데이터 (오늘의 가이드를 위함)
    last_row = df_all.iloc[-1]
    rsi_now = last_row['rsi']
    p1, p2 = last_row['prev_close'], last_row['prev_close2']
    
    # 오늘의 타점 계산
    x_raw = (p1 + p2) * 1.01 / 1.99
    willow_x = np.ceil(x_raw * 100) / 100
    
    if rsi_now > 65: mode, color, b_l, s_l = "Ivy", "red", willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
    elif rsi_now > 45: mode, color, b_l, s_l = "Willow", "orange", willow_x - 0.01, willow_x
    elif rsi_now > 30: mode, color, b_l, s_l = "Lily", "blue", np.floor((willow_x * 0.975) * 100) / 100, willow_x
    else: mode, color, b_l, s_l = "Tulip", "purple", np.floor((willow_x * 0.975) * 100) / 100, willow_x

    # --- 상단 현재 포트폴리오 상태 ---
    st.subheader(f"📊 현재 {target_ticker} 보유 현황")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("보유 수량", f"{int(current_shares)} 주")
    m2.metric("평균 단가", f"${current_avg:.2f}")
    m3.metric("진행 회차", f"{current_slots} / 5 슬롯")
    total_val = current_cash + (current_shares * last_row['close'])
    m4.metric("현재 총 자산", f"${total_val:,.2f}", f"{(total_val/init_seed - 1)*100:.2f}%")

    st.divider()

    # --- 오늘의 매매 가이드 ---
    st.markdown(f"### 🎯 오늘의 {target_ticker} 매매 가이드 (모드: :{color}[{mode}])")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.success(f"#### 📥 {current_slots + 1}회차 매수 예약 (LOC)")
        if current_slots < 5:
            # 다음 매수 시 시드 재계산 (현재 자산 기준)
            next_slot_cash = (current_cash + (current_shares * current_avg)) / 5
            buy_qty = int(next_slot_cash // b_l)
            st.write(f"**지정 가격:** `${b_l:.2f}` 이하")
            st.write(f"**매수 수량:** `{buy_qty}주` 권장")
        else:
            st.write("✅ 모든 슬롯이 채워졌습니다. 매도 시점을 기다리세요.")

    with col_r:
        st.error("#### 📤 전량 매도 예약 (LOC)")
        if current_shares > 0:
            st.write(f"**지정 가격:** `${s_l:.2f}` 이상")
            profit_rate = (s_l / current_avg - 1) * 100 if current_avg > 0 else 0
            st.write(f"**매도 시 수익률:** `{profit_rate:+.2f}%` (평단 대비)")
        else:
            st.write("보유 수량이 없습니다.")

    # --- 상세 지표 모니터링 ---
    with st.expander("🔍 상세 지표 확인"):
        c1, c2, c3 = st.columns(3)
        c1.metric("QQQ RSI (지표)", f"{rsi_now:.2f}")
        c2.metric("p1 (어제 종가)", f"${p1:.2f}")
        c3.metric("p2 (그저께 종가)", f"${p2:.2f}")
        st.write("현재까지의 자산 흐름:")
        # 히스토리 계산을 위해 리런 로직은 생략하고 간단한 차트만 표시
        st.line_chart(df_all['close'])

else:
    st.warning("데이터를 불러오지 못했습니다. 시작일이나 종목을 확인해주세요.")
