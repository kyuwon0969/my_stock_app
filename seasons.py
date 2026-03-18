import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 매니저 & 백테스터", page_icon="🌿", layout="wide")

# --- 데이터 엔진 ---
@st.cache_data(ttl=3600)
def get_processed_data(ticker, start_date, end_date):
    """지정한 기간의 데이터를 가져오고 지표 계산"""
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
    data = yf.download([ticker, "QQQ"], start=fetch_start, end=end_date, progress=False)
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
    
    # Wilder's RSI (QQQ 기준)
    delta = qqq_close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
    
    return df.loc[start_date:].dropna()

def run_simulation(df, initial_seed):
    """사계절 전략 시뮬레이션 엔진 (고정 슬롯 매수 로직)"""
    cash, shares, used_slots, slot_cash, avg_price = initial_seed, 0, 0, 0, 0
    history = []
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, rsi_val = row['prev_close'], row['prev_close2'], row['close'], row['rsi']
        x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # 모드 판정
        if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        # 1. 매도 (전량 매도 후 사이클 리셋)
        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            cash += (shares * curr_close)
            shares, used_slots, slot_cash, avg_price = 0, 0, 0, 0
            sold_today = True
        
        # 2. 매수 (고정 슬롯 방식)
        if not sold_today and used_slots < 5 and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / 5 # 사이클 시작 시점 1/5 고정
            
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        history.append({'Date': date, 'Total': cash + (shares * curr_close), 'Cash': cash, 'Slots': used_slots, 'Avg_Price': avg_price})
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
st.title("🌿 사계절 전략 통합 매니저")

with st.sidebar:
    st.header("⚙️ 기본 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD", "QLD"], index=0)
    st.divider()
    op_start = st.date_input("실제 운용 시작일", value=datetime(2024, 1, 1), key="op_start")
    init_seed = st.number_input("투자 원금 (USD)", value=10000.0, step=1000.0)

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

# --- TAB 1: 실시간 추적 ---
with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed)
        cur = hist_live.iloc[-1]
        last_data = df_live.iloc[-1]
        
        # 상단 현황
        st.subheader(f"📊 {target_ticker} 현재 운용 상태")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("보유 수량", f"{int(cur['Total']-cur['Cash'])//int(last_data['close'])} 주")
        m2.metric("평균 단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / 5 슬롯")
        m4.metric("현재 자산", f"${cur['Total']:,.2f}", f"{(cur['Total']/init_seed - 1)*100:.2f}%")

        # 주문 가이드
        st.divider()
        rsi_now = last_data['rsi']
        p1, p2 = last_data['prev_close'], last_data['prev_close2']
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_now > 65: mode, color, b_l, s_l = "Ivy", "red", willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_now > 45: mode, color, b_l, s_l = "Willow", "orange", willow_x - 0.01, willow_x
        elif rsi_now > 30: mode, color, b_l, s_l = "Lily", "blue", np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode, color, b_l, s_l = "Tulip", "purple", np.floor((willow_x * 0.975) * 100) / 100, willow_x

        st.markdown(f"### 🎯 오늘의 주문 가이드 (모드: :{color}[{mode}])")
        cl, cr = st.columns(2)
        with cl:
            st.success(f"#### 📥 {int(cur['Slots']) + 1}회차 매수 (LOC)")
            if cur['Slots'] < 5:
                st.write(f"**가격:** `${b_l:.2f}` 이하")
                st.caption(f"기준 RSI: {rsi_now:.2f} | p1: ${p1:.2f}")
            else: st.write("✅ 모든 슬롯 완료")
        with cr:
            st.error("#### 📤 전량 매도 (LOC)")
            if cur['Avg_Price'] > 0:
                st.write(f"**가격:** `${s_l:.2f}` 이상")
                st.write(f"**목표 수익률:** `{(s_l/cur['Avg_Price']-1)*100:+.2f}%`")
            else: st.write("보유 수량 없음")

        st.divider()
        st.subheader("📈 운용 이후 자산 흐름")
        st.line_chart(hist_live['Total'])

# --- TAB 2: 백테스트 ---
with tab2:
    st.header("🔍 과거 기간 성과 분석")
    col1, col2, col3 = st.columns(3)
    with col1: s_date = st.date_input("테스트 시작일", value=datetime(2016, 1, 1))
    with col2: e_date = st.date_input("테스트 종료일", value=datetime.now())
    with col3: s_seed = st.number_input("테스트 시드", value=10000)

    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, s_date.strftime('%Y-%m-%d'), e_date.strftime('%Y-%m-%d'))
        if df_back is not None:
            res_back = run_simulation(df_back, s_seed)
            final_v = res_back['Total'].iloc[-1]
            
            # 지표 계산
            total_ret = (final_v / s_seed - 1) * 100
            days = (res_back.index[-1] - res_back.index[0]).days
            cagr = ((final_v / s_seed) ** (365.25 / days) - 1) * 100
            mdd = (res_back['Total'] / res_back['Total'].cummax() - 1).min() * 100

            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("최종 자산", f"${final_v:,.2f}")
            r2.metric("총 수익률", f"{total_ret:,.2f}%")
            r3.metric("CAGR", f"{cagr:.2f}%")
            r4.metric("MDD", f"{mdd:.2f}%")

            st.line_chart(res_back['Total'])

            # 연도별 수익률 요약
            res_back['year'] = res_back.index.year
            yearly = []
            prev_v = s_seed
            for yr in res_back['year'].unique():
                y_df = res_back[res_back['year'] == yr]
                y_e = y_df['Total'].iloc[-1]
                yearly.append({'연도': yr, '수익률': f"{(y_e/prev_v-1)*100:.2f}%", 'MDD': f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.2f}%"})
                prev_v = y_e
            st.table(pd.DataFrame(yearly))
