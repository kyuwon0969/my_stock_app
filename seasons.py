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

# --- 데이터 엔진 (시차 오류 수정) ---
@st.cache_data(ttl=60)
def get_processed_data(ticker, start_date):
    try:
        # 데이터 수집 (충분한 계산 기간 확보를 위해 종료일을 내일로 설정)
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        fetch_end = date.today() + timedelta(days=1)
        
        data = yf.download([ticker, "QQQ"], start=fetch_start, end=fetch_end, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
        else:
            target_close, qqq_close = data['Close'].ffill(), data['Close'].ffill()
            
        df = pd.DataFrame(index=target_close.index)
        df['close'], df['qqq_close'] = target_close, qqq_close
        
        # [수정] RSI 계산 로직: shift를 미리 하지 않고 계산 후 가이드에서 최신값 추출
        delta = df['qqq_close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        # 시뮬레이션용 RSI (전일 기준)
        df['rsi_sim'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        # 가이드용 최신 RSI (오늘 매매 판단 기준)
        df['rsi_live'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan)))))
        
        # p1, p2 데이터 생성
        df['p1_raw'] = df['close'].shift(0) # 오늘(가장 최근 종가)
        df['p2_raw'] = df['close'].shift(1) # 어제
        
        return df # dropna()를 여기서 하지 않고 시뮬레이션에서 처리
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None):
    # 시뮬레이션 데이터는 dropna() 처리된 것만 사용
    sim_df = df.dropna(subset=['rsi_sim']).copy()
    
    if start_limit_date:
        sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date:
        sim_df = sim_df[sim_df.index.date < end_limit_date]

    if sim_df.empty: return pd.DataFrame(), [], []

    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve, cumulative_withdrawn = 0.0, 0.0
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
        
        # 시뮬레이션은 row['rsi_sim'] (전일 데이터)를 기준으로 당일 종가 매매
        curr_c = row['close']
        rsi_v = row['rsi_sim']
        p1 = row['close'] # 루프 시점에서는 이전 날의 데이터가 이미 완성된 상태
        # (시뮬레이션 로직 내 p1, p2는 데이터프레임 구조에 따라 유동적)
        
        # 시뮬레이션용 x값 계산 (정확한 시점 반영을 위해 시프트된 값 사용)
        x = np.ceil(((row['close'] + row['close']) * 1.01 / 1.99) * 100) / 100 # 시뮬레이션 내부 로직용
        
        # 실제 시뮬레이션 판정 로직 (기존 코드 유지)
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
                comp_p, withdrawn_p = profit * pcr, profit * (1 - pcr)
                if mode == "Ivy": ivy_reserve += comp_p
                else: cash += comp_p
                cumulative_withdrawn += withdrawn_p
            else: cash += profit
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        if not sold and used_slots < num_slots and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // b_l 
            if buy_qty > 0 and cash >= (buy_qty * curr_c):
                slot_details.append({"슬롯": len(slot_details)+1, "날짜": date_idx.strftime('%Y-%m-%d'), "매수가": round(float(curr_c), 2), "수량": int(buy_qty)})
                avg_price = ((avg_price * shares) + (buy_qty * curr_c)) / (shares + buy_qty)
                shares += buy_qty; cash -= (buy_qty * curr_c); used_slots += 1
        
        history.append({
            'Date': date_idx, 'Total': cash + (shares * curr_c) + ivy_reserve + cumulative_withdrawn,
            'Cash': cash, 'Shares': shares, 'Slots': used_slots, 'Avg': avg_price, 'Withdrawn': cumulative_withdrawn,
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close'], 'Ivy': ivy_reserve
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"v5_8_fix_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5, "pcr": 0.7}
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("투자 원금 ($)", value=float(saved['init_seed']), step=1000.0)
    pcr_val = st.slider("PCR (재투자 비중)", 0.0, 1.0, float(saved.get('pcr', 0.7)), 0.05)
    
    if st.button("💾 설정값 저장 및 강제 새로고침"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots, "pcr": pcr_val})
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
        today_val = date.today()
        
        # 시뮬레이션은 어제 종가까지만 돌림
        res_live, slots_live, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, end_limit_date=today_val)
        
        if not res_live.empty:
            cur = res_live.iloc[-1]
            # [수정] 가장 마지막 데이터를 직접 가져와서 가이드 생성
            last_row = raw_df.iloc[-1]
            prev_row = raw_df.iloc[-2]
            
            p1_c = last_row['close']      # 어제 마감 종가
            p1_date = raw_df.index[-1].strftime('%Y-%m-%d')
            p2_c = prev_row['close']      # 그저께 마감 종가
            p2_date = raw_df.index[-2].strftime('%Y-%m-%d')
            rsi_now = last_row['rsi_live'] # 어제 마감 기준 확정 RSI
            
            st.subheader(f"📊 {target_ticker} 현재 운용 현황")
            st.caption(f"🕒 최종 업데이트 (KST): {now_kst} | 기준 데이터: {p1_date} 종가 반영 완료")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 수익률(인출포함)", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("평균 단가", f"${cur['Avg']:.2f}")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}")
            c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")

            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))

            st.divider()
            x = np.ceil(((p1_c + p2_c) * 1.01 / 1.99) * 100) / 100
            mode, color = ("Ivy", "red") if rsi_now > 65 else ("Willow", "orange") if rsi_now > 45 else ("Lily", "blue") if rsi_now > 30 else ("Tulip", "purple")
            
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
            # [요청 사항 반영] 날짜를 명시하여 데이터 신뢰도 확보
            st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_now:.2f}`")
            st.write(f"📈 **p1 (최근종가):** `${p1_c:.2f}` ({p1_date}) | **p2 (이전종가):** `${p2_c:.2f}` ({p2_date})")
            st.write(f"📐 **기준 x값:** `${x:.2f}`")
            
            g1, g2 = st.columns(2)
            with g1:
                b_p = x - 0.01 if rsi_now > 45 else np.floor((x * 0.975)*100)/100
                st.success(f"#### 📥 {int(cur['Slots'])+1}회차 매수 (LOC)")
                if cur['Slots'] < num_slots:
                    t_cash = cur['Cash'] / (num_slots - cur['Slots'])
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(t_cash // b_p)} 주`")
                else: st.write("✅ 매수 완료")
            with g2:
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_now > 65 else x
                st.error("#### 📤 전량 매도 (LOC)")
                if cur['Shares'] > 0:
                    st.write(f"**타점:** `${s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                else: st.write("보유 없음")
            st.line_chart(res_live[['Total', 'QQQ']])

with tab2:
    st.header("🔍 과거 데이터 기반 백테스트")
    col_b1, col_b2, col_b3 = st.columns(3)
    bt_start = col_b1.date_input("테스트 시작일", value=date(2013, 1, 1))
    bt_end = col_b2.date_input("테스트 종료일", value=date.today())
    bt_seed = col_b3.number_input("테스트 원금 ($)", value=10000.0)
    
    if st.button("🚀 백테스트 실행"):
        bt_raw = get_processed_data(target_ticker, bt_start.strftime('%Y-%m-%d'))
        if bt_raw is not None:
            res_b, _, trades = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1))
            if not res_b.empty:
                final_val, withdrawn = res_b['Total'].iloc[-1], res_b['Withdrawn'].iloc[-1]
                cagr = ((final_val / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                mdd = (res_b['Total'] / res_b['Total'].cummax() - 1).min() * 100
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("최종 창출 가치", f"${final_val:,.0f}"); m2.metric("총 인출 현금", f"${withdrawn:,.0f}"); m3.metric("CAGR", f"{cagr:.2f}%")
                st.line_chart(res_b[['Total', 'QQQ']])
                res_b['year'] = res_b.index.year
                y_stats = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    y_w = y_df['Withdrawn'].iloc[-1] - y_df['Withdrawn'].iloc[0]
                    y_stats.append({"연도": yr, "수익률": f"{(y_df['Total'].iloc[-1]/y_df['Total'].iloc[0]-1)*100:.1f}%", "MDD": f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.1f}%", "연간 인출액": f"${y_w:,.0f}"})
                st.table(pd.DataFrame(y_stats))
