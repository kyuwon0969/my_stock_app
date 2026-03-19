import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [BOXX 이자 실험]", page_icon="🌿", layout="wide")

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

def run_simulation(df, initial_seed, num_slots):
    """Ivy 수익 저축 & BOXX 운용(이자 5.3%) & Tulip 투입 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve = 0.0  
    
    # BOXX 로직: 연 5.3% 수익률을 일복리로 환산 (영업일 252일 기준)
    boxx_annual_rate = 0.053 
    daily_boxx_rate = (1 + boxx_annual_rate) ** (1/252) - 1
    
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p1, p2 = row['prev_close'], row['prev_close2']
        curr_close, qqq_curr_close, rsi_val = row['close'], row['qqq_close'], row['rsi']
        
        # [추가] BOXX 이자 정산: 저축액이 있다면 매일 이자가 붙음
        if ivy_reserve > 0:
            ivy_reserve *= (1 + daily_boxx_rate)
        
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_val > 65: mode = "Ivy"; b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: mode = "Willow"; b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: mode = "Lily"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode = "Tulip"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            sell_proceeds = shares * curr_close
            profit = sell_proceeds - (avg_price * shares)
            if mode == "Ivy" and profit > 0:
                cash += (avg_price * shares) # 원금 회수
                ivy_reserve += profit        # 수익금은 BOXX로 저축
            else:
                cash += sell_proceeds        # 일반 복리 회수
            shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
            sold_today = True
        
        # [핵심] Tulip 진입 시 BOXX(이자 포함) 전액을 본체 시드로 합산
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve
            ivy_reserve = 0.0
            
        if not sold_today and used_slots < num_slots and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        total_assets = cash + (shares * curr_close) + ivy_reserve
        qqq_hold_val = (initial_seed / qqq_start_price) * qqq_curr_close
        
        history.append({
            'Date': date, 'Total': float(total_assets), 'QQQ_Hold': float(qqq_hold_val),
            'Ivy_Reserve': float(ivy_reserve), 'Cash': float(cash), 'Shares': float(shares), 
            'Slots': int(used_slots), 'Avg_Price': float(avg_price), 'Slot_Cash': float(slot_cash)
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 전략 및 운용 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"seasons_boxx_exp_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("설정 저장 완료!")
    if st.button("🔄 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

# --- TAB 1: 실시간 추적 & 가이드 ---
with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots)
        cur, last = hist_live.iloc[-1], df_live.iloc[-1]
        
        st.subheader(f"📊 {target_ticker} 운용 현황 (Ivy 수익 BOXX 운용 중)")
        m1, m2, m3, m4 = st.columns(4)
        total_ret, qqq_ret = (cur['Total'] / init_seed - 1) * 100, (cur['QQQ_Hold'] / init_seed - 1) * 100
        m1.metric("전략 수익률", f"{total_ret:+.2f}%", f"QQQ 대비 {total_ret-qqq_ret:+.2f}%")
        m2.metric("평균 단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / {num_slots} 슬롯")
        m4.metric("현재 총 자산", f"${cur['Total']:,.2f}")
        
        # BOXX 저축액 표시 (매일 이자가 붙음)
        st.info(f"🏦 현재 Ivy 비상금(BOXX) 저축액: **${cur['Ivy_Reserve']:,.2f}** (연 5.3% 일복리 적용 중)")

        st.divider()
        rsi_now, p1, p2 = last['rsi'], last['prev_close'], last['prev_close2']
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_now > 65: mode, color, b_l, s_l = "Ivy", "red", willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_now > 45: mode, color, b_l, s_l = "Willow", "orange", willow_x - 0.01, willow_x
        elif rsi_now > 30: mode, color, b_l, s_l = "Lily", "blue", np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode, color, b_l, s_l = "Tulip", "purple", np.floor((willow_x * 0.975) * 100) / 100, willow_x

        st.markdown(f"### 🎯 오늘의 실전 주문 가이드 (현재 모드: :{color}[{mode}])")
        st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_now:.2f}` | 전일 종가(p1) `${p1:.2f}` | 전전일 종가(p2) `${p2:.2f}`")
        
        cl, cr = st.columns(2)
        with cl:
            st.success(f"#### 📥 {int(cur['Slots']) + 1}회차 매수 (LOC)")
            if int(cur['Slots']) < num_slots:
                target_slot_cash = cur['Cash'] / (num_slots - int(cur['Slots'])) if int(cur['Slots']) == 0 else cur['Slot_Cash']
                buy_qty = int(target_slot_cash // b_l)
                st.write(f"**매수 가격:** `${b_l:.2f}` 이하 (LOC)")
                st.write(f"**권장 수량:** `{buy_qty} 주` (${target_slot_cash:,.2f} 규모)")
            else: st.write("✅ 모든 슬롯 매수 완료 (풀매수 상태)")
        with cr:
            st.error("#### 📤 전량 매도 (LOC)")
            if cur['Shares'] > 0:
                st.write(f"**매도 가격:** `${s_l:.2f}` 이상 (LOC)")
                st.write(f"**매도 수량:** `{int(cur['Shares'])} 주` (전량)")
                st.write(f"**목표 수익률:** `{(s_l/cur['Avg_Price']-1)*100:+.2f}%` (평단 대비)")
            else: st.write("보유 물량 없음")
            
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])

# --- TAB 2: 백테스트 ---
with tab2:
    st.header(f"🔍 {num_slots}슬롯 Ivy 저축(BOXX 이자) 백테스트")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("테스트 시작일", value=datetime(2011, 1, 1), key="bt_s")
    with c2: e_date = st.date_input("테스트 종료일", value=datetime.now(), key="bt_e")
    with c3: s_seed = st.number_input("테스트 시드", value=10000.0, step=1000.0, key="bt_seed")

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
                r3.metric("CAGR (연복리)", f"{cagr:.2f}%")
                r4.metric("전체 MDD", f"{mdd:.2f}%")
                
                st.line_chart(res_back[['Total', 'QQQ_Hold']])

                # 연도별 성과 상세 요약
                res_back['year'] = res_back.index.year
                yearly_data = []
                temp_seed = s_seed
                for yr in sorted(res_back['year'].unique()):
                    y_df = res_back[res_back['year'] == yr]
                    y_e_v = y_df['Total'].iloc[-1]
                    q_start_v = y_df['QQQ_Hold'].iloc[0]
                    q_end_v = y_df['QQQ_Hold'].iloc[-1]
                    q_r_y = (q_end_v / q_start_v - 1) * 100
                    y_mdd = (y_df['Total'] / y_df['Total'].cummax() - 1).min() * 100
                    
                    yearly_data.append({
                        '연도': yr, 
                        '전략 수익률': f"{(y_e_v/temp_seed-1)*100:.2f}%", 
                        'QQQ 수익률': f"{q_r_y:.2f}%", 
                        '연도별 MDD': f"{y_mdd:.2f}%",
                        '최대 Ivy 저축액(BOXX)': f"${y_df['Ivy_Reserve'].max():,.0f}"
                    })
                    temp_seed = y_e_v
                st.subheader("📅 연도별 성과 리포트 (BOXX 이자 반영)")
                st.table(pd.DataFrame(yearly_data))
