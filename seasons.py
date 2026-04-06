import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from datetime import datetime, date, timedelta
import pytz
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 Pro", page_icon="🌿", layout="wide")
localS = LocalStorage()
KST = pytz.timezone('Asia/Seoul')

# --- 데이터 엔진 ---
@st.cache_data(ttl=300)
def get_processed_data(ticker, start_date):
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        fetch_end = date.today() + timedelta(days=1)
        
        data = yf.download([ticker, "QQQ"], start=fetch_start, end=fetch_end, progress=False)
        if data.empty or 'Close' not in data: 
            return None
        
        close_df = data['Close']
        if ticker in close_df.columns: target_close = close_df[ticker].ffill()
        else: target_close = close_df.iloc[:, 0].ffill()
            
        if "QQQ" in close_df.columns: qqq_close = close_df['QQQ'].ffill()
        else: qqq_close = target_close 

        df = pd.DataFrame(index=target_close.index)
        df['close'], df['qqq_close'] = target_close, qqq_close
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        df['rsi_live'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan)))))
        
        return df.dropna(subset=['p2_c'])
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None, pending_dep=0.0, manual_withdrawn=0.0):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.copy()
    if start_limit_date:
        sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date:
        sim_df = sim_df[sim_df.index.date < end_limit_date]
    
    # [안정화] 만약 데이터가 비어있다면 초기 상태를 담은 1행짜리 데이터프레임 반환
    if sim_df.empty:
        last_row = df.iloc[-1]
        history = [{
            'Date': datetime.now(), 'Total': float(initial_seed), 'Cash': float(initial_seed), 
            'Shares': 0.0, 'Slots': 0, 'Avg': 0.0, 'Withdrawn': 0.0, 'Ivy': 0.0,
            'QQQ_Price': last_row['qqq_close'], 'Target_Price': last_row['close']
        }]
        return pd.DataFrame(history).set_index('Date'), [], []

    cash = float(initial_seed) - float(manual_withdrawn)
    shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
    ivy_reserve, cumulative_withdrawn = 0.0, 0.0 
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
        if np.isnan(rsi_v): continue
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        sold = False
        if shares > 0 and curr_c >= s_l:
            total_sell_val = shares * curr_c
            profit = total_sell_val - (avg_price * shares)
            trade_profits.append(profit)
            cash += (avg_price * shares)
            if profit > 0:
                comp_profit = profit * pcr
                if mode == "Ivy": ivy_reserve += comp_profit
                else: cash += comp_profit
                cumulative_withdrawn += profit * (1 - pcr)
            else: cash += profit
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        if used_slots == 0 and float(pending_dep) != 0:
            cash += float(pending_dep); pending_dep = 0.0

        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        if not sold and used_slots < (num_slots + 1) and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            current_order_cash = cash if used_slots >= num_slots else slot_cash
            buy_qty = current_order_cash // b_l 
            if buy_qty > 0 and cash >= (buy_qty * curr_c):
                actual_cost = buy_qty * curr_c
                slot_details.append({"슬롯": "예비" if used_slots >= num_slots else used_slots + 1, "날짜": date_idx.strftime('%Y-%m-%d'), "매수가(종가)": round(float(curr_c), 2), "수량": int(buy_qty), "금액": round(float(actual_cost), 2)})
                avg_price = ((avg_price * shares) + actual_cost) / (shares + buy_qty)
                shares += buy_qty; cash -= actual_cost; used_slots += 1
        
        history.append({
            'Date': date_idx, 'Total': cash + (shares * curr_c) + ivy_reserve + cumulative_withdrawn,
            'Cash': cash, 'Shares': shares, 'Slots': used_slots, 'Avg': avg_price, 
            'Withdrawn': cumulative_withdrawn, 'Ivy': ivy_reserve,
            'QQQ_Price': row['qqq_close'], 'Target_Price': row['close']
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 및 저장 로직 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = "SOXL"
    config_key = f"v5_pro_final_{target_ticker}"
    q_params = st.query_params
    saved_ls = localS.getItem(config_key) or {}
    
    def get_setting(key, default):
        val = q_params.get(key) or saved_ls.get(key)
        return val if val is not None else default

    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(get_setting('num_slots', 5)))
    op_start_val = get_setting('op_start', "2024-01-01")
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(op_start_val).date())
    init_seed = st.number_input("투자 원금 ($)", value=float(get_setting('init_seed', 10000.0)), step=1000.0)
    pcr_val = st.slider("PCR (재투자 비중)", 0.0, 1.0, float(get_setting('pcr', 0.7)), 0.05)
    p_dep = st.number_input("추가 입금/출금액 ($)", value=float(get_setting('pending_dep', 0.0)))

    if st.button("💾 설정값 저장 및 강제 새로고침"):
        params = {"num_slots": num_slots, "op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "pcr": pcr_val, "pending_dep": p_dep}
        st.query_params.update(params)
        localS.setItem(config_key, params)
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트", "📖 Info (도움말)"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
        res_live, slots_live, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, pending_dep=p_dep)
        
        if not res_live.empty:
            cur = res_live.iloc[-1]
            # 사이클 기준금액 계산 (이력이 없으면 현재 원금)
            zero_slots_df = res_live[res_live['Slots'] == 0]
            base_capital = zero_slots_df.iloc[-1]['Cash'] if not zero_slots_df.empty else init_seed
            fund_cycle = base_capital + p_dep

            st.subheader(f"📊 {target_ticker} 현재 운용 현황")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("총 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("평균 단가", f"${cur['Avg']:.2f}")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}" if cur['Slots'] <= num_slots else f"{num_slots} + 예비")
            c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")
            c5.metric("사이클 기준금액", f"${fund_cycle:,.2f}")
            
            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))
            st.divider()
            
            latest_data = raw_df.iloc[-1]
            prev_data = raw_df.iloc[-2]
            rsi_val = latest_data['rsi_live']
            x = np.ceil(((latest_data['close'] + prev_data['close']) * 1.01 / 1.99) * 100) / 100
            mode, color = ("Ivy", "red") if rsi_val > 65 else ("Willow", "orange") if rsi_val > 45 else ("Lily", "blue") if rsi_val > 30 else ("Tulip", "purple")
            
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
            g1, g2 = st.columns(2)
            with g1:
                b_p = x - 0.01 if rsi_val > 45 else np.floor((x * 0.975)*100)/100
                if cur['Slots'] < num_slots:
                    st.error(f"#### {int(cur['Slots'])+1}회차 정규 매수 (LOC)")
                    order_cash = (fund_cycle + (cur['Ivy'] if mode == "Tulip" and cur['Slots'] == 0 else 0)) / num_slots
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(order_cash // b_p)} 주`")
                elif cur['Slots'] == num_slots:
                    st.warning(f"#### 🔥 예비 슬롯 추가 매수 (LOC)")
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(cur['Cash'] // b_p)} 주`")
                else: st.write("✅ 매수 완료")
            with g2:
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                st.info(f"#### 📤 전량 매도 (LOC)")
                if cur['Shares'] > 0: st.write(f"**타점:** `${s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                else: st.write("보유 없음")
            
            chart_data = res_live.copy()
            chart_data['QQQ'] = (init_seed / chart_data['QQQ_Price'].iloc[0]) * chart_data['QQQ_Price']
            chart_data[target_ticker] = (init_seed / chart_data['Target_Price'].iloc[0]) * chart_data['Target_Price']
            st.line_chart(chart_data[['Total', 'QQQ', target_ticker]])
        else:
            st.warning("⚠️ 데이터를 시뮬레이션할 수 없습니다. 시작일을 조금 더 과거로 설정해 주세요.")

with tab2:
    st.header("🔍 과거 데이터 기반 백테스트")
    col_b1, col_b2, col_b3 = st.columns(3)
    bt_start = col_b1.date_input("테스트 시작일", value=date(2013, 1, 1), key="bt_s")
    bt_end = col_b2.date_input("테스트 종료일", value=date.today(), key="bt_e")
    bt_seed = col_b3.number_input("테스트 원금 ($)", value=10000.0, key="bt_v")
    
    if st.button("🚀 백테스트 실행"):
        bt_raw = get_processed_data(target_ticker, bt_start.strftime('%Y-%m-%d'))
        if bt_raw is not None:
            res_b, _, trades = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1))
            if not res_b.empty:
                f_val = res_b['Total'].iloc[-1]
                cagr = ((f_val / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                mdd = (res_b['Total'] / res_b['Total'].cummax() - 1).min() * 100
                total_sells = len(trades)
                win_rate = (len([t for t in trades if t > 0]) / total_sells * 100) if total_sells > 0 else 0
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("최종 자산", f"${f_val:,.0f}"); m2.metric("CAGR (연복리)", f"{cagr:.2f}%"); m3.metric("MDD", f"{mdd:.2f}%")
                s1, s2, s3 = st.columns(3)
                s1.metric("총 매도 횟수", f"{total_sells}회"); s2.metric("승률", f"{win_rate:.1f}%"); s3.metric("총 인출 현금", f"${res_b['Withdrawn'].iloc[-1]:,.0f}")
                res_b['QQQ_Norm'] = (bt_seed / res_b['QQQ_Price'].iloc[0]) * res_b['QQQ_Price']
                res_b[f'{target_ticker}_Hold'] = (bt_seed / res_b['Target_Price'].iloc[0]) * res_b['Target_Price']
                st.line_chart(res_b[['Total', 'QQQ_Norm', f'{target_ticker}_Hold']])
                y_stats = [{"연도": yr, "수익률": f"{(y_df['Total'].iloc[-1]/y_df['Total'].iloc[0]-1)*100:.1f}%", "MDD": f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.1f}%"} for yr, y_df in res_b.groupby(res_b.index.year)]
                st.table(pd.DataFrame(y_stats))

with tab3:
    st.header("📖 이용 가이드")
    st.markdown("전략 모드와 오늘의 실전 가이드를 확인하여 매일 밤 LOC 주문을 예약하세요.")
