import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm, skew, kurtosis
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [10일 지연 복리]", page_icon="🌿", layout="wide")

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
        df['close'], df['qqq_close'] = target_close, qqq_close
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots, delay_days=10):
    """10일 지연 복리 및 BOXX 이자 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    pending_profits = [] # (금액, 남은일수)
    boxx_rate = (1 + 0.053) ** (1/252) - 1
    history = []
    qqq_start_p = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        # 1. 지연 수익금 정산 & 이자 가산
        ready_to_merge, new_pending = 0.0, []
        for amt, days in pending_profits:
            accrued = amt * (1 + boxx_rate)
            if days <= 1: ready_to_merge += accrued
            else: new_pending.append((accrued, days - 1))
        pending_profits = new_pending
        cash += ready_to_merge
        
        # 2. 타점 산출
        x = np.ceil(((row['p1_c'] + row['p2_c']) * 1.01 / 1.99) * 100) / 100
        rsi = row['rsi']
        if rsi > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        # 3. 매매 로직
        sold = False
        if shares > 0 and row['close'] >= s_l:
            profit = (shares * row['close']) - (avg_price * shares)
            cash += (avg_price * shares)
            if profit > 0: pending_profits.append((profit, delay_days))
            else: cash += profit # 손실은 즉시 반영
            shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
            sold = True
            
        if not sold and used_slots < num_slots and row['close'] <= b_l:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // row['close']
            if buy_qty > 0 and cash >= (buy_qty * row['close']):
                avg_price = ((avg_price * shares) + (buy_qty * row['close'])) / (shares + buy_qty)
                shares += buy_qty; cash -= (buy_qty * row['close']); used_slots += 1
        
        pend_total = sum(p[0] for p in pending_profits)
        total = cash + (shares * row['close']) + pend_total
        history.append({
            'Date': date, 'Total': total, 'Pending': pend_total, 'Cash': cash, 
            'Shares': shares, 'Slots': used_slots, 'Avg_Price': avg_price,
            'QQQ_Hold': (initial_seed / qqq_start_p) * row['qqq_close']
        })
    return pd.DataFrame(history).set_index('Date')

# --- 분석 엔진 ---
def analyze_results(df_res, initial_seed):
    returns = df_res['Total'].pct_change().dropna()
    ann_ret = (df_res['Total'].iloc[-1] / initial_seed) ** (252 / len(df_res)) - 1
    
    # Sharpe/Sortino
    rf_daily = (1 + 0.053)**(1/252) - 1
    excess = returns - rf_daily
    sharpe = np.sqrt(252) * excess.mean() / returns.std()
    sortino = np.sqrt(252) * excess.mean() / returns[returns < 0].std()
    
    # MDD/Recovery
    peak = df_res['Total'].cummax()
    dd = (df_res['Total'] - peak) / peak
    mdd = dd.min()
    
    durations, curr = [], 0
    for d in (dd < 0):
        if d: curr += 1
        else:
            if curr > 0: durations.append(curr)
            curr = 0
    max_rec = max(durations) if durations else 0

    # DSR (보수적 비교군 10개 설정)
    T, sk, kr = len(returns), skew(returns), kurtosis(returns)
    sigma_sr = np.sqrt((1 + 0.5 * sk**2 + (kr-3)/4 * sharpe**2) / (T-1))
    dsr = norm.cdf((sharpe - np.sqrt(2 * np.log(10))) / sigma_sr)

    return {
        "CAGR": ann_ret * 100, "Sharpe": sharpe, "Sortino": sortino,
        "MDD": mdd * 100, "Recovery": max_rec, "DSR": dsr, "Final": df_res['Total'].iloc[-1]
    }

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 전략 및 지연 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    delay_days = st.slider("수익금 지연 일수", 1, 20, 10)
    config_key = f"seasons_delayed_v2_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("슬롯 분할", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("운용 시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("원금 (USD)", value=float(saved['init_seed']), step=1000.0)
    
    if st.button("💾 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")

tab1, tab2 = st.tabs(["🎯 실전 가이드", "📊 전문 백테스트 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        res_live = run_simulation(df_live, init_seed, num_slots, delay_days)
        cur, last = res_live.iloc[-1], df_live.iloc[-1]
        
        st.subheader(f"📊 {target_ticker} 운용 현황 ({delay_days}일 지연 복리)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전략 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
        m2.metric("지연 수익금", f"${cur['Pending']:,.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} 슬롯")
        m4.metric("현재 자산", f"${cur['Total']:,.2f}")

        st.divider()
        # 가이드 및 데이터 출력
        rsi_now, p1, p2 = last['rsi'], last['p1_c'], last['p2_c']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        st.write(f"🔍 **판단 데이터:** RSI `{rsi_now:.2f}` | p1 `${p1:.2f}` | p2 `${p2:.2f}`")
        
        cl, cr = st.columns(2)
        with cl:
            st.success("#### 📥 매수 가이드")
            if cur['Slots'] < num_slots:
                st.write(f"가격: `${x-0.01:.2f}` 이하 (LOC)")
            else: st.write("✅ 풀 매수 완료")
        with cr:
            st.error("#### 📤 매도 가이드")
            if cur['Shares'] > 0:
                st.write(f"가격: `${x:.2f}` 이상 (LOC)")
            else: st.write("보유 물량 없음")
        st.line_chart(res_live[['Total', 'QQQ_Hold']])

with tab2:
    st.header(f"🔍 {target_ticker} 정밀 리스크 분석")
    if st.button("🚀 전체 백테스트 실행"):
        df_back = get_processed_data(target_ticker, "2013-01-01")
        if df_back is not None:
            res_back = run_simulation(df_back, init_seed, num_slots, delay_days)
            m = analyze_results(res_back, init_seed)
            
            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("CAGR (연복리)", f"{m['CAGR']:.2f}%")
            r2.metric("MDD (최대낙폭)", f"{m['MDD']:.2f}%")
            r3.metric("최장 회복 기간", f"{m['Recovery']}일")
            r4.metric("DSR (신뢰도)", f"{m['DSR']:.4f}")
            
            r5, r6, r7, r8 = st.columns(4)
            r5.metric("Sharpe Ratio", f"{m['Sharpe']:.3f}")
            r6.metric("Sortino Ratio", f"{m['Sortino']:.3f}")
            r7.metric("최종 자산", f"${m['Final']:,.2f}")
            r8.info("💡 DSR > 0.5 이면 우수")

            st.line_chart(res_back[['Total', 'QQQ_Hold']])
            
            # 연도별 표
            res_back['year'] = res_back.index.year
            yearly = []
            temp_seed = init_seed
            for yr in sorted(res_back['year'].unique()):
                y_df = res_back[res_back['year'] == yr]
                y_val = y_df['Total'].iloc[-1]
                y_mdd = (y_df['Total'] / y_df['Total'].cummax() - 1).min() * 100
                yearly.append({'연도': yr, '수익률': f"{(y_val/temp_seed-1)*100:.2f}%", 'MDD': f"{y_mdd:.2f}%"})
                temp_seed = y_val
            st.table(pd.DataFrame(yearly))
