import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from datetime import datetime, date, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 Pro (KRW)", page_icon="🌿", layout="wide")
localS = LocalStorage()

# --- 데이터 엔진 (환율 데이터 추가) ---
@st.cache_data(ttl=300)
def get_processed_data(ticker, start_date):
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        # 메인 종목, QQQ, 환율(USDKRW) 데이터 수집
        data = yf.download([ticker, "QQQ", "USDKRW=X"], start=fetch_start, progress=False)
        
        # SCHD 데이터 및 배당 정보 수집
        schd_ticker = yf.Ticker("SCHD")
        schd_raw = schd_ticker.history(start=fetch_start, actions=True)
        
        if data.empty or schd_raw.empty: return None
        
        # 데이터 구조 정리 (MultiIndex 대응)
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
            fx_rate = data['Close']['USDKRW=X'].ffill()
        else:
            target_close, qqq_close, fx_rate = data['Close'].ffill(), data['Close'].ffill(), data['Close'].ffill()
            
        df = pd.DataFrame(index=target_close.index)
        df['close'], df['qqq_close'], df['fx'] = target_close, qqq_close, fx_rate
        
        # SCHD 및 배당금 매칭 (시간대 제거)
        df.index = df.index.tz_localize(None)
        schd_raw.index = schd_raw.index.tz_localize(None)
        df['schd_close'] = schd_raw['Close'].reindex(df.index, method='ffill')
        df['schd_div'] = schd_raw['Dividends'].reindex(df.index).fillna(0)
        
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 (원화 환산 로직 추가) ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.copy()
    if start_limit_date: sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date: sim_df = sim_df[sim_df.index.date < end_limit_date]
    if sim_df.empty: return pd.DataFrame(), [], []

    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve, schd_shares = 0.0, 0.0
    cumulative_div_usd, cumulative_div_krw = 0.0, 0.0 # [추가] 달러/원화 배당금 누적
    pending_schd_buy = 0.0 
    
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        # 1. 자산 정산
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
        
        # [원화 환산 추가] 배당금 발생 시점의 환율로 원화 금액 계산
        if schd_shares > 0 and row['schd_div'] > 0:
            div_today_usd = schd_shares * row['schd_div']
            cumulative_div_usd += div_today_usd
            cumulative_div_krw += div_today_usd * row['fx'] # 당일 환율 적용
            
        if pending_schd_buy > 0:
            schd_shares += pending_schd_buy / row['schd_close']
            pending_schd_buy = 0.0
            
        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        sold = False
        if shares > 0 and curr_c >= s_l:
            profit = (shares * curr_c) - (avg_price * shares)
            trade_profits.append(profit)
            cash += (avg_price * shares)
            if profit > 0:
                comp_p = profit * pcr
                if mode == "Ivy": ivy_reserve += comp_p
                else: cash += comp_p
                pending_schd_buy = profit * (1 - pcr)
            else: cash += profit
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        if not sold and used_slots < num_slots and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // b_l 
            actual_cost = buy_qty * curr_c
            if buy_qty > 0 and cash >= actual_cost:
                slot_details.append({
                    "슬롯": len(slot_details) + 1, "날짜": date_idx.strftime('%Y-%m-%d'),
                    "매수가": round(float(curr_c), 2), "수량": int(buy_qty)
                })
                avg_price = ((avg_price * shares) + actual_cost) / (shares + buy_qty)
                shares += buy_qty; cash -= actual_cost; used_slots += 1
        
        schd_val = schd_shares * row['schd_close']
        total_usd = cash + (shares * curr_c) + ivy_reserve + schd_val + cumulative_div_usd
        
        history.append({
            'Date': date_idx, 'Total': total_usd, 'Total_KRW': total_usd * row['fx'], # [추가] 총 자산 원화 환산
            'Cash': cash, 'Shares': shares, 'Slots': used_slots, 'Avg': avg_price, 
            'SCHD_Val': schd_val, 'Div_USD': cumulative_div_usd, 'Div_KRW': cumulative_div_krw,
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close'], 'Ivy': ivy_reserve, 'FX': row['fx']
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"v8_5_krw_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("투자 원금 ($)", value=float(saved['init_seed']), step=1000.0)
    pcr_val = st.slider("PCR (수익 재투자 비율)", 0.0, 1.0, 0.7, 0.05)
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")

tab1, tab2 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        today_val = date.today()
        res_live, slots_live, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, end_limit_date=today_val)
        
        if not res_live.empty:
            cur, last = res_live.iloc[-1], raw_df.iloc[-1]
            st.subheader(f"📊 {target_ticker} 운용 현황 (환율: {cur['FX']:,.1f}원)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 수익률 (USD)", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("현재 총 자산 (KRW)", f"{int(cur['Total_KRW']):,}원")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}")
            c4.metric("누적 배당금 (KRW)", f"{int(cur['Div_KRW']):,}원")
            
            sc1, sc2 = st.columns(2)
            sc1.info(f"🏦 Ivy 비상금(BOXX): **${cur['Ivy']:,.2f}**")
            sc2.warning(f"💰 누적 배당금(USD): **${cur['Div_USD']:,.2f}**")

            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))

            st.divider()
            rsi_n = last['rsi']
            mode, color = ("Ivy", "red") if rsi_n > 65 else ("Willow", "orange") if rsi_n > 45 else ("Lily", "blue") if rsi_n > 30 else ("Tulip", "purple")
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
            st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_n:.2f}` | 현재 환율 `{cur['FX']:,.1f}` | 기준 $x$ `{np.ceil(((last['p1_c']+last['p2_c'])*1.01/1.99)*100)/100:.2f}`")
            
            p1, p2 = last['p1_c'], last['p2_c']
            x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
            g1, g2 = st.columns(2)
            with g1:
                b_p = x - 0.01 if rsi_n > 45 else np.floor((x * 0.975)*100)/100
                st.success(f"#### 📥 다음 매수 타점: `${b_p:.2f}` (LOC)")
            with g2:
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_n > 65 else x
                st.error("#### 📤 다음 매도 타점: `${s_p:.2f}` (LOC)")
            st.line_chart(res_live[['Total', 'QQQ']])

with tab2:
    st.header("🔍 과거 데이터 기반 백테스트 (원화/배당 분석)")
    col_b1, col_b2, col_b3 = st.columns(3)
    bt_start, bt_end = col_b1.date_input("시작", value=date(2013, 1, 1)), col_b2.date_input("종료", value=date.today())
    bt_seed = col_b3.number_input("시드 ($)", value=10000.0)
    
    if st.button("🚀 백테스트 실행"):
        bt_raw = get_processed_data(target_ticker, bt_start.strftime('%Y-%m-%d'))
        if bt_raw is not None:
            res_b, _, trades = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1))
            if not res_b.empty:
                f_v_usd, f_v_krw = res_b['Total'].iloc[-1], res_b['Total_KRW'].iloc[-1]
                cagr = ((f_v_usd / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                mdd = (res_b['Total'] / res_b['Total'].cummax() - 1).min() * 100

                st.subheader("🏆 백테스트 종합 리포트")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("최종 자산 (KRW)", f"{int(f_v_krw):,}원")
                m2.metric("CAGR (USD)", f"{cagr:.2f}%")
                m3.metric("전체 MDD", f"{mdd:.2f}%")
                m4.metric("총 배당금 (KRW)", f"{int(res_b['Div_KRW'].iloc[-1]):,}원")
                
                st.line_chart(res_b[['Total', 'QQQ']])
                res_b['year'] = res_b.index.year
                y_stats = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    y_stats.append({"연도": yr, "수익률(USD)": f"{(y_df['Total'].iloc[-1]/y_df['Total'].iloc[0]-1)*100:.1f}%", "누적 배당(KRW)": f"{int(y_df['Div_KRW'].iloc[-1]):,}원", "SCHD 비중": f"${y_df['SCHD_Val'].iloc[-1]:,.0f}"})
                st.table(pd.DataFrame(y_stats))
