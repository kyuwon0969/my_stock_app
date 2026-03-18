import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [이평선 실험 모드]", page_icon="📈", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 (이동평균선 추가) ---
@st.cache_data(ttl=600)
def get_processed_data(ticker, start_date):
    try:
        # 이평선 계산을 위해 시작일 1년 전부터 수집
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
        data = yf.download([ticker, "QQQ"], start=fetch_start, progress=False)
        
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
        else:
            target_close = data['Close'].ffill()
            qqq_close = data['Close'].ffill()
        
        df = pd.DataFrame(index=target_close.index)
        df['close'] = target_close
        df['qqq_close'] = qqq_close
        df['prev_close'] = df['close'].shift(1)
        df['prev_close2'] = df['close'].shift(2)
        
        # --- [실험 로직] 이동평균선 계산 (QQQ 기준) ---
        df['sma20'] = qqq_close.rolling(window=20).mean().shift(1)
        df['sma60'] = qqq_close.rolling(window=60).mean().shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots):
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, qqq_curr_close = row['prev_close'], row['prev_close2'], row['close'], row['qqq_close']
        sma20, sma60 = row['sma20'], row['sma60']
        
        x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # --- [실험 로직] 이평선 기반 모드 판정 ---
        # 골든크로스 상태 (20 > 60): 공격 모드
        if sma20 > sma60:
            # 강세 중에서도 이격도가 크면 Ivy, 아니면 Willow (임시 구분)
            if qqq_curr_close > sma20 * 1.05: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
            else: b_limit, s_limit = willow_x - 0.01, willow_x
        # 데드크로스 상태 (20 <= 60): 방어 모드
        else:
            if qqq_curr_close < sma20 * 0.95: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
            else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            cash += (shares * curr_close)
            shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
            sold_today = True
        
        if not sold_today and used_slots < num_slots and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        total_assets = cash + (shares * curr_close)
        qqq_hold_val = (initial_seed / qqq_start_price) * qqq_curr_close
        
        history.append({
            'Date': date, 'Total': float(total_assets), 'QQQ_Hold': float(qqq_hold_val),
            'QQQ_Price': float(qqq_curr_close), 'Cash': float(cash), 'Shares': float(shares), 
            'Slots': int(used_slots), 'Avg_Price': float(avg_price), 'Slot_Cash': float(slot_cash),
            'sma20': sma20, 'sma60': sma60
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 (TAB 1 가이드 부분 수정) ---
with st.sidebar:
    st.header("⚙️ 실험: 이평선 크로스 모드")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"seasons_exp_config_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    op_start = st.date_input("운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    if st.button("💾 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("실험 설정 저장!")

tab1, tab2 = st.tabs(["🎯 실시간 추적", "📊 백테스트 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots)
        cur, last = hist_live.iloc[-1], df_live.iloc[-1]
        
        # 모드 판정 표시
        is_golden = last['sma20'] > last['sma60']
        mode_text = "🚀 강세 (골든크로스)" if is_golden else "🛡️ 약세 (데드크로스)"
        mode_color = "red" if is_golden else "blue"

        st.subheader(f"📊 {target_ticker} 현황 | 모드: :{mode_color}[{mode_text}]")
        st.write(f"현재 QQQ 20일선: `${last['sma20']:.2f}` | 60일선: `${last['sma60']:.2f}`")
        
        m1, m2, m3, m4 = st.columns(4)
        total_ret, qqq_ret = (cur['Total'] / init_seed - 1) * 100, (cur['QQQ_Hold'] / init_seed - 1) * 100
        m1.metric("전략 수익률", f"{total_ret:+.2f}%", f"QQQ 대비 {total_ret-qqq_ret:+.2f}%")
        m2.metric("평균 단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / {num_slots}")
        m4.metric("현재 자산", f"${cur['Total']:,.2f}")
        
        st.divider()
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])
