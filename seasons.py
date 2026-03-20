import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from datetime import datetime, date, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 매니저 V6", page_icon="🌿", layout="wide")
localS = LocalStorage()

# --- 데이터 엔진 ---
@st.cache_data(ttl=300)
def get_processed_data(ticker, start_date):
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        data = yf.download([ticker, "QQQ"], start=fetch_start, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
        else:
            target_close, qqq_close = data['Close'].ffill(), data['Close'].ffill()
            
        df = pd.DataFrame(index=target_close.index)
        df['close'], df['qqq_close'] = target_close, qqq_close
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 (보험금 수령 로직 추가) ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.copy()
    if start_limit_date: sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date: sim_df = sim_df[sim_df.index.date < end_limit_date]
    if sim_df.empty: return pd.DataFrame(), [], []

    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve = 0.0
    safe_cash = 0.0 # PCR로 모인 30%의 안전 자산
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
            
        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        # 모드 판정
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        # 1. 매도 로직
        sold = False
        if shares > 0 and curr_c >= s_l:
            total_sell_val = shares * curr_c
            profit = total_sell_val - (avg_price * shares)
            trade_profits.append(profit)
            
            # 원금 회수
            cash += (avg_price * shares)
            
            if profit > 0:
                # [익절 시] PCR 적용하여 수익 배분
                compounding_profit = profit * pcr
                withdrawn_profit = profit * (1 - pcr)
                
                if mode == "Ivy": ivy_reserve += compounding_profit
                else: cash += compounding_profit
                
                safe_cash += withdrawn_profit # 30%는 보험금으로 저축
            else:
                # [손절 시] 보험금 수령 로직 발동!
                # 손실은 100% 캐시에서 깎고, 그동안 모은 safe_cash를 전액 수혈
                cash += profit # 손실 반영 (음수값 더하기)
                
                # 보험금 투입 (소방수!)
                if safe_cash > 0:
                    cash += safe_cash
                    # 시뮬레이션 기록을 위해 기록 남기기 가능
                    safe_cash = 0.0 
            
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        # 2. Tulip 진입 시 Ivy 비상금 투입
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        # 3. 매수 로직 (정량 매수)
        if not sold and used_slots < num_slots and curr_c <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // b_l 
            actual_cost = buy_qty * curr_c
            if buy_qty > 0 and cash >= actual_cost:
                slot_details.append({
                    "슬롯": len(slot_details) + 1, "날짜": date_idx.strftime('%Y-%m-%d'),
                    "매수가(종가)": round(float(curr_c), 2), "기준가(타점)": round(float(b_l), 2),
                    "수량": int(buy_qty), "금액": round(float(actual_cost), 2)
                })
                avg_price = ((avg_price * shares) + actual_cost) / (shares + buy_qty)
                shares += buy_qty; cash -= actual_cost; used_slots += 1
        
        total_assets = cash + (shares * curr_c) + ivy_reserve + safe_cash
        history.append({
            'Date': date_idx, 'Total': total_assets, 'Cash': cash, 'Shares': shares, 
            'Slots': used_slots, 'Avg': avg_price, 'Safe_Cash': safe_cash,
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close'], 'Ivy': ivy_reserve
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"v6_final_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("투자 원금 ($)", value=float(saved['init_seed']), step=1000.0)
    pcr_val = st.slider("PCR (수익 재투자 비율)", 0.0, 1.0, 0.7, 0.05)
    
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("설정이 저장되었습니다!")

tab1, tab2 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        today_val = date.today()
        res_live, slots_live, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, end_limit_date=today_val)
        
        if not res_live.empty:
            cur = res_live.iloc[-1]
            last_actual = raw_df.iloc[-1] 
            
            st.subheader(f"📊 {target_ticker} 현재 운용 현황")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("평균 단가", f"${cur['Avg']:.2f}")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}")
            c4.metric("현재 총 자산", f"${cur['Total']:,.2f}")
            
            sc1, sc2 = st.columns(2)
            sc1.info(f"🏦 Ivy 비상금(BOXX): **${cur['Ivy']:,.2f}**")
            sc2.success(f"🛡️ 안전 현금(보험금): **${cur['Safe_Cash']:,.2f}**")

            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))

            st.divider()
            rsi_n = last_actual['rsi']
            mode, color = ("Ivy", "red") if rsi_n > 65 else ("Willow", "orange") if rsi_n > 45 else ("Lily", "blue") if rsi_n > 30 else ("Tulip", "purple")
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
            st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_n:.2f}` | p1 `${last_actual['p1_c']:.2f}` | p2 `${last_actual['p2_c']:.2f}`")
            
            # 매매 가이드 (생략 없음)
            p1, p2 = last_actual['p1_c'], last_actual['p2_c']
            x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
            g1, g2 = st.columns(2)
            with g1:
                b_p = x - 0.01 if rsi_n > 45 else np.floor((x * 0.975)*100)/100
                st.success(f"#### 📥 {int(cur['Slots'])+1}회차 매수 (LOC)")
                if cur['Slots'] < num_slots:
                    t_cash = cur['Cash'] / (num_slots - cur['Slots'])
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(t_cash // b_p)} 주`")
                else: st.write("✅ 매수 완료")
            with g2:
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_n > 65 else x
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
                final_val = res_b['Total'].iloc[-1]
                total_ret, days = (final_val / bt_seed - 1) * 100, (res_b.index[-1] - res_b.index[0]).days
                cagr = ((final_val / bt_seed) ** (365.25 / days) - 1) * 100
                peak = res_b['Total'].cummax()
                dd = (res_b['Total'] - peak) / peak
                mdd, avg_dd = dd.min() * 100, dd[dd < 0].mean() * 100
                
                if trades:
                    wins = [t for t in trades if t > 0]
                    win_rate = (len(wins) / len(trades)) * 100
                    pl_ratio = np.mean(wins) / abs(np.mean([t for t in trades if t <= 0])) if len(trades) > len(wins) else float('inf')
                else: win_rate, pl_ratio = 0, 0

                st.divider()
                st.subheader("🏆 백테스트 종합 결과 리포트")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("최종 자산", f"${final_val:,.0f}")
                m2.metric("총 수익률", f"{total_ret:.1f}%")
                m3.metric("CAGR", f"{cagr:.2f}%")
                m4.metric("승률", f"{win_rate:.1f}%")
                
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("MDD", f"{mdd:.2f}%")
                m6.metric("Average DD", f"{avg_dd:.2f}%")
                m7.metric("손익비", f"{pl_ratio:.2f}")
                m8.metric("잔여 보험금", f"${res_b['Safe_Cash'].iloc[-1]:,.0f}")

                st.line_chart(res_b[['Total', 'QQQ']])
                
                # 연도별 상세 테이블
                res_b['year'] = res_b.index.year
                yearly_stats = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    y_ret = (y_df['Total'].iloc[-1] / y_df['Total'].iloc[0] - 1) * 100
                    y_mdd = (y_df['Total'] / y_df['Total'].cummax() - 1).min() * 100
                    yearly_stats.append({"연도": yr, "수익률": f"{y_ret:.1f}%", "MDD": f"{y_mdd:.1f}%", "안전현금(누적)": f"${y_df['Safe_Cash'].iloc[-1]:,.0f}"})
                st.table(pd.DataFrame(yearly_stats))
