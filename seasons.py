import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [10일 지연 복리 실험]", page_icon="🌿", layout="wide")

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
            target_close = data['Close'].ffill()
            qqq_close = data['Close'].ffill()
            
        df = pd.DataFrame(index=target_close.index)
        df['close'] = target_close
        df['qqq_close'] = qqq_close
        df['prev_close'] = df['close'].shift(1)
        df['prev_close2'] = df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots, delay_days=10):
    """익절 후 N거래일 대기 후 본체 합류 (지연 복리 엔진)"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    
    # [지연 복리 관리] (금액, 남은 일수) 리스트
    pending_profits = [] 
    
    # BOXX 이자 (연 5.3%)
    boxx_annual_rate = 0.053 
    daily_boxx_rate = (1 + boxx_annual_rate) ** (1/252) - 1
    
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p1, p2 = row['prev_close'], row['prev_close2']
        curr_close, qqq_curr_close, rsi_val = row['close'], row['qqq_close'], row['rsi']
        
        # 1. 지연 수익금 정산 & 이자 계산
        new_pending = []
        ready_to_merge = 0.0
        
        for amount, days_left in pending_profits:
            # BOXX 이자 적용
            accrued_amount = amount * (1 + daily_boxx_rate)
            if days_left <= 1:
                ready_to_merge += accrued_amount
            else:
                new_pending.append((accrued_amount, days_left - 1))
        
        pending_profits = new_pending
        cash += ready_to_merge # 10일이 지난 수익금 본체 합류
        
        # 2. x값 및 모드 판정
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        # 3. 매도 로직
        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            sell_proceeds = shares * curr_close
            profit = sell_proceeds - (avg_price * shares)
            
            # 원금은 즉시 회수, 수익금만 지연 주머니에 넣음
            cash += (avg_price * shares)
            if profit > 0:
                pending_profits.append((profit, delay_days))
            else:
                cash += profit # 손실은 즉시 시드에 반영
                
            shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
            sold_today = True
        
        # 4. 매수 로직
        if not sold_today and used_slots < num_slots and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        current_pending_total = sum(p[0] for p in pending_profits)
        total_assets = cash + (shares * curr_close) + current_pending_total
        qqq_hold_val = (initial_seed / qqq_start_price) * qqq_curr_close
        
        history.append({
            'Date': date, 'Total': float(total_assets), 'Pending': current_pending_total,
            'Cash': float(cash), 'Shares': float(shares), 'Slots': int(used_slots), 
            'Avg_Price': float(avg_price)
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 전략 및 운용 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD"], index=0)
    delay_input = st.number_input("복리 지연 일수 (거래일)", value=10, min_value=1)
    
    config_key = f"seasons_delayed_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("지연 복리 설정 저장 완료!")

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots, delay_input)
        cur = hist_live.iloc[-1]
        
        st.subheader(f"📊 {target_ticker} 운용 현황 ({delay_input}일 지연 복리)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전략 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
        m2.metric("지연 중인 수익", f"${cur['Pending']:,.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} 슬롯")
        m4.metric("현재 총 자산", f"${cur['Total']:,.2f}")
        
        st.info(f"⌛ 현재 **${cur['Pending']:,.2f}**의 수익금이 {delay_input}거래일 대기 리스트에 있습니다. (BOXX 이자 발생 중)")
        st.line_chart(hist_live[['Total']])

with tab2:
    st.header(f"🔍 {delay_input}일 지연 복리 백테스트")
    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, "2013-01-01")
        if df_back is not None:
            res_back = run_simulation(df_back, init_seed, num_slots, delay_input)
            
            f_v = res_back['Total'].iloc[-1]
            t_r = (f_v / init_seed - 1) * 100
            mdd = (res_back['Total'] / res_back['Total'].cummax() - 1).min() * 100
            
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("최종 자산", f"${f_v:,.2f}")
            r2.metric("총 수익률", f"{t_r:.2f}%")
            r3.metric("전체 MDD", f"{mdd:.2f}%")
            
            st.line_chart(res_back['Total'])
