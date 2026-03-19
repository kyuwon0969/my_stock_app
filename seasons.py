import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [Tulip 단리 실험]", page_icon="🌿", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 (기존 오리지널과 동일) ---
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

def run_simulation(df, initial_seed, num_slots):
    """Tulip 단리 & Lily 증액 로직 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash = float(initial_seed)
    shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
    
    # [실험 변수] Tulip 수익 예비금
    tulip_reserve = 0.0
    
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p1, p2 = row['prev_close'], row['prev_close2']
        curr_close, qqq_curr_close, rsi_val = row['close'], row['qqq_close'], row['rsi']
        
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # 모드 판정
        if rsi_val > 65: mode = "Ivy"; b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: mode = "Willow"; b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: mode = "Lily"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode = "Tulip"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            sell_proceeds = shares * curr_close
            profit = sell_proceeds - (avg_price * shares)
            
            # [핵심 실험 로직]
            if mode == "Tulip":
                # Tulip일 때는 원금만 회수하고 수익금은 예비금으로 격리 (단리 효과)
                cash += (avg_price * shares) 
                tulip_reserve += profit
            elif mode == "Lily":
                # Lily일 때는 수익금을 포함해 회수하고, 쌓인 예비금까지 합쳐서 증액
                cash += sell_proceeds + tulip_reserve
                tulip_reserve = 0.0
            else:
                # Ivy, Willow는 일반 복리
                cash += sell_proceeds
                
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
        
        total_assets = cash + (shares * curr_close) + tulip_reserve
        history.append({
            'Date': date, 'Total': float(total_assets), 'Reserve': float(tulip_reserve),
            'Cash': float(cash), 'Shares': float(shares), 'Slots': int(used_slots), 
            'Avg_Price': float(avg_price), 'Mode': mode
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 (생략 없이 통합) ---
with st.sidebar:
    st.header("⚙️ Tulip 단리 실험 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"seasons_tulip_exp_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("슬롯 수", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    op_start = st.date_input("시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("원금", value=float(saved_config['init_seed']))
    if st.button("💾 실험 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("실험 로직 저장 완료!")

tab1, tab2 = st.tabs(["🎯 실전 추적", "📊 실험 백테스트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots)
        cur = hist_live.iloc[-1]
        st.subheader(f"📊 {target_ticker} 현황 (예비금: ${cur['Reserve']:,.2f})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 자산", f"${cur['Total']:,.2f}")
        m2.metric("예비 현금", f"${cur['Reserve']:,.2f}", help="Tulip 수익 보관함")
        m3.metric("현재 모드", cur['Mode'])
        m4.metric("진행 슬롯", f"{int(cur['Slots'])}회")
        st.line_chart(hist_live[['Total']])

with tab2:
    st.header("🔍 Tulip 단리 vs Lily 증액 성과 분석")
    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, "2013-01-01")
        if df_back is not None:
            res_back = run_simulation(df_back, init_seed, num_slots)
            st.line_chart(res_back[['Total']])
            # 성과 지표 생략 없이 연도별 표 출력 로직...
            res_back['year'] = res_back.index.year
            yearly = []
            temp_seed = init_seed
            for yr in sorted(res_back['year'].unique()):
                y_df = res_back[res_back['year'] == yr]
                y_e_v = y_df['Total'].iloc[-1]
                yearly.append({'연도': yr, '수익률': f"{(y_e_v/temp_seed-1)*100:.2f}%", '최대 예비금': f"${y_df['Reserve'].max():,.0f}"})
                temp_seed = y_e_v
            st.table(pd.DataFrame(yearly))
