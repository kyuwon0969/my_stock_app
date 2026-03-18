import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [볼린저 밴드 실험]", page_icon="🔮", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 (볼린저 밴드 추가) ---
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
        
        # [실험] QQQ 기준 볼린저 밴드 (20일, 2표준편차)
        ma20 = qqq_close.rolling(window=20).mean()
        std20 = qqq_close.rolling(window=20).std()
        df['bb_mid'] = ma20.shift(1)
        df['bb_up'] = (ma20 + (std20 * 2)).shift(1)
        df['bb_low'] = (ma20 - (std20 * 2)).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots):
    """볼린저 밴드 기반 시뮬레이션 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, qqq_curr_close = row['prev_close'], row['prev_close2'], row['close'], row['qqq_close']
        bb_low, bb_mid, bb_up = row['bb_low'], row['bb_mid'], row['bb_up']
        
        x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # [실험 판정] 주가가 밴드 어디에 위치하느냐에 따라 모드 변경
        if qqq_curr_close < bb_low:
            # 과매도 상태: 보수적으로 아주 싸게 매수 시도
            b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        elif qqq_curr_close > bb_mid:
            # 상승 추세 혹은 중심선 위: 공격적 매수 타점
            b_limit, s_limit = willow_x - 0.01, willow_x
        else:
            # 밴드 중간 영역: 일반 타점
            b_limit, s_limit = willow_x - 0.01, willow_x

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
            'Slots': int(used_slots), 'Avg_Price': float(avg_price), 'Slot_Cash': float(slot_cash)
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 실험: 볼린저 밴드 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"seasons_exp_bb_config_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    op_start = st.date_input("운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    
    if st.button("💾 실험 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("볼린저 밴드 설정 저장!")
    if st.button("🔄 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots)
        cur, last = hist_live.iloc[-1], df_live.iloc[-1]
        
        # 밴드 위치에 따른 텍스트 표시
        if last['qqq_close'] < last['bb_low']: bb_pos = "🔵 하단 밴드 이탈 (매수 기회)"
        elif last['qqq_close'] > last['bb_up']: bb_pos = "🔴 상단 밴드 돌파 (과열)"
        else: bb_pos = "🟢 밴드 내부 횡보"

        st.subheader(f"📊 {target_ticker} 현황 | 상태: {bb_pos}")
        st.write(f"현재 QQQ 볼린저 하단: `${last['bb_low']:.2f}` | 상단: `${last['bb_up']:.2f}`")
        
        m1, m2, m3, m4 = st.columns(4)
        total_ret, qqq_ret = (cur['Total'] / init_seed - 1) * 100, (cur['QQQ_Hold'] / init_seed - 1) * 100
        m1.metric("전략 수익률", f"{total_ret:+.2f}%", f"QQQ 대비 {total_ret-qqq_ret:+.2f}%")
        m2.metric("평균 단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / {num_slots}")
        m4.metric("현재 자산", f"${cur['Total']:,.2f}")
        
        st.divider()
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])

with tab2:
    st.header(f"🔍 {num_slots}슬롯 볼린저 밴드 성과 분석")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("테스트 시작일", value=datetime(2013, 1, 1), key="bb_bt_s")
    with c2: e_date = st.date_input("테스트 종료일", value=datetime.now(), key="bb_bt_e")
    with c3: s_seed = st.number_input("테스트 시드", value=10000.0, step=1000.0, key="bb_bt_seed")

    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, s_date.strftime('%Y-%m-%d'))
        if df_back is not None:
            df_back = df_back.loc[:e_date.strftime('%Y-%m-%d')]
            res_back = run_simulation(df_back, s_seed, num_slots)
            
            if not res_back.empty:
                f_v, f_q = res_back['Total'].iloc[-1], res_back['QQQ_Hold'].iloc[-1]
                t_r, q_r = (f_v / s_seed - 1) * 100, (f_q / s_seed - 1) * 100
                days = (res_back.index[-1] - res_back.index[0]).days
                cagr = ((f_v / s_seed) ** (365.25 / (days if days > 0 else 1)) - 1) * 100
                mdd = (res_back['Total'] / res_back['Total'].cummax() - 1).min() * 100

                st.divider()
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("최종 자산", f"${f_v:,.2f}")
                r2.metric("총 수익률", f"{t_r:,.2f}%", f"QQQ 대비 {t_r-q_r:+.2f}%")
                r3.metric("CAGR", f"{cagr:.2f}%")
                r4.metric("전략 MDD", f"{mdd:.2f}%")
                st.line_chart(res_back[['Total', 'QQQ_Hold']])

                # 연도별 표
                res_back['year'] = res_back.index.year
                yearly_data = []
                temp_seed = s_seed
                for yr in sorted(res_back['year'].unique()):
                    y_df = res_back[res_back['year'] == yr]
                    y_e_v = y_df['Total'].iloc[-1]
                    q_p = df_back['qqq_close'][df_back.index.year == yr]
                    q_r_y = (q_p.iloc[-1] / q_p.iloc[0] - 1) * 100
                    yearly_data.append({'연도': yr, '전략 수익률': f"{(y_e_v/temp_seed-1)*100:.2f}%", 'QQQ 수익률': f"{q_r_y:.2f}%", '전략 MDD': f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.2f}%"})
                    temp_seed = y_e_v
                st.subheader("📅 연도별 성과 (볼린저 밴드)")
                st.table(pd.DataFrame(yearly_data))
