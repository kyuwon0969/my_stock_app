import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
from datetime import datetime, date, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 Pro V9 (KRW)", page_icon="🌿", layout="wide")
localS = LocalStorage()

# --- 데이터 엔진 (환율 & SCHD 배당 & 병합 로직 강화) ---
@st.cache_data(ttl=600)
def get_processed_data(ticker, start_date):
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        # 1. 메인 종목 및 QQQ
        main_data = yf.download([ticker, "QQQ", "USDKRW=X"], start=fetch_start, progress=False)
        
        # 2. SCHD (배당금을 위해 별도 수집)
        schd_ticker = yf.Ticker("SCHD")
        schd_raw = schd_ticker.history(start=fetch_start, actions=True)
        
        if main_data.empty or schd_raw.empty:
            return None

        # 데이터 구조 정리 (MultiIndex 처리)
        if isinstance(main_data.columns, pd.MultiIndex):
            df = pd.DataFrame(index=main_data.index)
            df['close'] = main_data['Close'][ticker].ffill()
            df['qqq_close'] = main_data['Close']['QQQ'].ffill()
            df['fx'] = main_data['Close']['USDKRW=X'].ffill()
        else:
            df = main_data[['Close']].rename(columns={'Close': 'close'}) # 단일 종목 시 처리

        # SCHD 종가 및 배당금 매칭 (인덱스 정규화)
        schd_raw.index = schd_raw.index.tz_localize(None) # 시간대 제거
        df.index = df.index.tz_localize(None)
        
        df['schd_close'] = schd_raw['Close'].reindex(df.index, method='ffill')
        df['schd_div'] = schd_raw['Dividends'].reindex(df.index).fillna(0)
        
        # 보조 지표 계산
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        delta = df['qqq_close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.copy()
    if start_limit_date: sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date: sim_df = sim_df[sim_df.index.date < end_limit_date]
    if sim_df.empty: return pd.DataFrame(), [], []

    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve, schd_shares, cumulative_div = 0.0, 0.0, 0.0
    pending_schd_buy = 0.0
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        # 1. 자산 정산
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
        if schd_shares > 0 and row['schd_div'] > 0:
            cumulative_div += schd_shares * row['schd_div']
        if pending_schd_buy > 0:
            schd_shares += pending_schd_buy / row['schd_close']
            pending_schd_buy = 0.0

        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        # 2. 매도
        sold = False
        if shares > 0 and curr_c >= s_l:
            profit = (shares * curr_c) - (avg_price * shares)
            trade_profits.append(profit)
            cash += (avg_price * shares)
            if profit > 0:
                ivy_reserve += (profit * pcr) if mode == "Ivy" else 0
                if mode != "Ivy": cash += (profit * pcr)
                pending_schd_buy = profit * (1 - pcr)
            else: cash += profit
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        # 3. 비상금 합류
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        # 4. 매수 (정량)
        if not sold and used_slots < num_slots and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // b_l 
            if buy_qty > 0 and cash >= (buy_qty * curr_c):
                slot_details.append({"슬롯": len(slot_details)+1, "날짜": date_idx.strftime('%Y-%m-%d'), "매수가": round(float(curr_c), 2), "수량": int(buy_qty)})
                avg_price = ((avg_price * shares) + (buy_qty * curr_c)) / (shares + buy_qty)
                shares += buy_qty; cash -= (buy_qty * curr_c); used_slots += 1
        
        total_usd = cash + (shares * curr_c) + ivy_reserve + (schd_shares * row['schd_close']) + cumulative_div
        history.append({
            'Date': date_idx, 'Total_USD': total_usd, 'Total_KRW': total_usd * row['fx'],
            'Ivy': ivy_reserve, 'Div': cumulative_div, 'FX': row['fx'], 'SCHD_Val': schd_shares * row['schd_close'],
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close']
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목", ["SOXL", "USD"], index=0)
    config_key = f"v9_krw_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("슬롯", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("원금 ($)", value=float(saved['init_seed']), step=1000.0)
    pcr_val = st.slider("PCR (재투자 비율)", 0.0, 1.0, 0.7, 0.05)
    if st.button("💾 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")

tab1, tab2 = st.tabs(["🎯 실시간 현황", "📊 과거 데이터 백테스트"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        today_v = date.today()
        res_l, slots_l, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, end_limit_date=today_v)
        if not res_l.empty:
            cur, last = res_l.iloc[-1], raw_df.iloc[-1]
            st.subheader(f"📊 {target_ticker} 운용 현황 (환율: {cur['FX']:,.1f}원)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("수익률 (USD)", f"{(cur['Total_USD']/init_seed-1)*100:+.2f}%")
            c2.metric("현재 자산 (KRW)", f"{int(cur['Total_KRW']):,}원")
            c3.metric("채워진 슬롯", f"{int(len(slots_l))} / {num_slots}")
            c4.metric("누적 배당 ($)", f"${cur['Div']:,.2f}")
            
            if slots_l: st.table(pd.DataFrame(slots_l))

            st.divider()
            rsi_n = last['rsi']
            mode, color = ("Ivy", "red") if rsi_n > 65 else ("Willow", "orange") if rsi_n > 45 else ("Lily", "blue") if rsi_n > 30 else ("Tulip", "purple")
            st.markdown(f"### 🎯 오늘의 가이드 (현재 모드: :{color}[{mode}])")
            st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_n:.2f}` | 환율 `{cur['FX']:,.1f}` | x값 `{np.ceil(((last['p1_c']+last['p2_c'])*1.01/1.99)*100)/100:.2f}`")
            st.line_chart(res_l[['Total_USD', 'QQQ']])

with tab2:
    st.header("🔍 과거 데이터 기반 백테스트 (원화/배당 분석)")
    col_b1, col_b2, col_b3 = st.columns(3)
    bt_start, bt_end = col_b1.date_input("시작", value=date(2013, 1, 1)), col_b2.date_input("종료", value=date.today())
    bt_seed = col_b3.number_input("시드 ($)", value=10000.0)
    
    if st.button("🚀 백테스트 실행"):
        bt_raw = get_processed_data(target_ticker, bt_start.strftime('%Y-%m-%d'))
        if bt_raw is not None:
            res_b, _, _ = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1))
            if not res_b.empty:
                final_usd = res_b['Total_USD'].iloc[-1]
                cagr = ((final_usd / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                mdd = (res_b['Total_USD'] / res_b['Total_USD'].cummax() - 1).min() * 100

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("최종 자산 (KRW)", f"{int(res_b['Total_KRW'].iloc[-1]):,}원")
                m2.metric("CAGR (USD)", f"{cagr:.2f}%")
                m3.metric("전체 MDD", f"{mdd:.2f}%")
                m4.metric("총 배당 ($)", f"${res_b['Div'].iloc[-1]:,.2f}")

                # 연도별 리포트 (연배당 포함)
                res_b['year'] = res_b.index.year
                y_data = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    # 해당 연도 발생 배당금 계산
                    y_div = y_df['Div'].iloc[-1] - y_df['Div'].iloc[0]
                    y_ret = (y_df['Total_USD'].iloc[-1] / y_df['Total_USD'].iloc[0] - 1) * 100
                    y_fx = y_df['FX'].mean()
                    y_data.append({"연도": yr, "수익률(USD)": f"{y_ret:.1f}%", "연배당($)": f"${y_div:,.2f}", "연배당(KRW)": f"{int(y_div * y_fx):,}원", "평균환율": f"{y_fx:.1f}"})
                
                st.markdown("#### 📅 연도별 성과 및 배당 리포트")
                st.table(pd.DataFrame(y_data))
                st.line_chart(res_b[['Total_USD', 'QQQ']])
