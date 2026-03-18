import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 멀티 매니저", page_icon="🌿", layout="wide")

localS = LocalStorage()

# --- 데이터 수집 함수 ---
@st.cache_data(ttl=300)
def fetch_live_data(ticker):
    data = yf.download([ticker, "QQQ"], period="6mo", progress=False)
    if data.empty or len(data) < 20: return None
    
    if isinstance(data.columns, pd.MultiIndex):
        target_close = data['Close'][ticker].dropna()
        qqq_close = data['Close']['QQQ'].dropna()
    else:
        target_close = data[ticker].dropna()
        qqq_close = data['QQQ'].dropna()

    today = datetime.now().strftime('%Y-%m-%d')
    if target_close.index[-1].strftime('%Y-%m-%d') == today:
        p_live, p1, p2 = float(target_close.iloc[-1]), float(target_close.iloc[-2]), float(target_close.iloc[-3])
        qqq_for_rsi = qqq_close.iloc[:-1]
    else:
        p_live = p1 = float(target_close.iloc[-1])
        p2 = float(target_close.iloc[-2])
        qqq_for_rsi = qqq_close

    delta = qqq_for_rsi.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    return float(rsi.iloc[-1]), p1, p2, p_live

@st.cache_data(ttl=3600)
def get_backtest_data(ticker, start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
    data = yf.download([ticker, "QQQ"], start=fetch_start, end=end_date, progress=False)
    if data.empty: return None
    
    if isinstance(data.columns, pd.MultiIndex):
        target_close = data['Close'][ticker].dropna()
        qqq_close = data['Close']['QQQ'].dropna()
    else:
        target_close = data[ticker].dropna()
        qqq_close = data['QQQ'].dropna()
    
    df = pd.DataFrame(index=target_close.index)
    df['close'] = target_close
    df['prev_close'] = df['close'].shift(1)
    df['prev_close2'] = df['close'].shift(2)
    
    delta = qqq_close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
    
    return df.loc[start_date:].dropna()

# --- UI 레이아웃 ---
st.title("🌿 사계절 전략 멀티 매니저")

with st.sidebar:
    st.header("⚙️ 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD", "QLD"], index=0)
    
    st.divider()
    user_name = st.text_input("사용자 이름", value="규원")
    storage_key = f"seasons_{target_ticker}_{user_name}"
    saved_data = localS.getItem(storage_key) or {"seed": 10000.0, "profit": 0.0, "slot": 0}
    
    init_seed = st.number_input(f"초기 시드 (USD)", value=float(saved_data['seed']))
    current_profit = st.number_input(f"누적 수익금 (USD)", value=float(saved_data['profit']))
    current_slot = st.slider("현재 매수 완료 회차", 0, 5, int(saved_data['slot']))
    
    if st.button("💾 데이터 저장"):
        localS.setItem(storage_key, {"seed": init_seed, "profit": current_profit, "slot": current_slot})
        st.success(f"데이터 저장 완료!")

tab1, tab2 = st.tabs([f"🎯 {target_ticker} 실시간 가이드", "📊 과거 백테스트"])

# --- TAB 1: 실시간 가이드 ---
with tab1:
    res = fetch_live_data(target_ticker)
    if res:
        rsi, p1, p2, live = res
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi > 65: mode, color, b_l, s_l = "Ivy", "red", willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi > 45: mode, color, b_l, s_l = "Willow", "orange", willow_x - 0.01, willow_x
        elif rsi > 30: mode, color, b_l, s_l = "Lily", "blue", np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode, color, b_l, s_l = "Tulip", "purple", np.floor((willow_x * 0.975) * 100) / 100, willow_x

        st.markdown(f"### {target_ticker} 현재 모드: :{color}[{mode}]")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QQQ RSI (지표)", f"{rsi:.2f}")
        c2.metric("p1 (어제)", f"${p1:.2f}")
        c3.metric("p2 (그저께)", f"${p2:.2f}")
        c4.metric(f"현재가", f"${live:.2f}", delta=f"{live-p1:.2f}")

        st.divider()
        total_cap = init_seed + current_profit
        one_slot = total_cap / 5
        buy_qty = int(one_slot // b_l) if b_l > 0 else 0
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.success(f"#### 📥 {current_slot + 1}회차 매수 (LOC)")
            if current_slot < 5:
                st.write(f"**매수 가격:** `${b_l:.2f}` 이하")
                st.write(f"**매수 수량:** `{buy_qty}주` 권장")
                st.caption(f"1슬롯 예산: ${one_slot:,.2f}")
            else: st.write("✅ 모든 슬롯 체결 완료")
        with col_r:
            st.error("#### 📤 전량 매도 (LOC)")
            st.write(f"**매도 가격:** `${s_l:.2f}` 이상")
            st.write(f"**목표 수익:** {((s_l/willow_x)-1)*100:+.1f}% (x값 대비)")

# --- TAB 2: 백테스트 ---
with tab2:
    st.header(f"📈 {target_ticker} 전략 성과 분석")
    col_a, col_b, col_c = st.columns(3)
    with col_a: b_start = st.date_input("시작일", value=datetime(2023, 1, 1))
    with col_b: b_end = st.date_input("종료일", value=datetime.now())
    with col_c: b_seed = st.number_input("테스트 시드 (USD)", value=10000, step=1000)
    
    if st.button(f"🚀 {target_ticker} 백테스트 실행"):
        df_back = get_backtest_data(target_ticker, b_start.strftime('%Y-%m-%d'), b_end.strftime('%Y-%m-%d'))
        if df_back is not None:
            cash, shares, used_slots, slot_cash = b_seed, 0, 0, 0
            history = []
            for date, row in df_back.iterrows():
                p_prev1, p_prev2, curr_close, rsi_val = row['prev_close'], row['prev_close2'], row['close'], row['rsi']
                x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
                willow_x = np.ceil(x_raw * 100) / 100
                if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
                elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
                elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
                else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
                
                sold_today = False
                if shares > 0 and curr_close >= s_limit:
                    cash += (shares * curr_close); shares, used_slots, slot_cash, sold_today = 0, 0, 0, True
                if not sold_today and used_slots < 5 and curr_close <= b_limit:
                    if used_slots == 0: slot_cash = cash / 5
                    if cash >= slot_cash:
                        buy_qty = slot_cash // curr_close
                        shares += buy_qty; cash -= (buy_qty * curr_close); used_slots += 1
                history.append({'Date': date, 'Total': cash + (shares * curr_close)})
            
            res_df = pd.DataFrame(history).set_index('Date')
            final_val = res_df['Total'].iloc[-1]
            total_ret, days = (final_val / b_seed - 1) * 100, (res_df.index[-1] - res_df.index[0]).days
            cagr = ((final_val / b_seed) ** (365.25 / days) - 1) * 100
            mdd = (res_df['Total'] / res_df['Total'].cummax() - 1).min() * 100

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("최종 자산", f"${final_val:,.2f}")
            m2.metric("총 수익률", f"{total_ret:,.2f}%")
            m3.metric("CAGR", f"{cagr:.2f}%")
            m4.metric("최대 낙폭(MDD)", f"{mdd:.2f}%")
            
            st.subheader("📅 연도별 성과 요약")
            res_df['year'] = res_df.index.year
            prev_v = b_seed
            summary = []
            for yr in res_df['year'].unique():
                y_df_yr = res_df[res_df['year'] == yr]
                y_e = y_df_yr['Total'].iloc[-1]
                summary.append({'연도': yr, '수익률': f"{(y_e/prev_v-1)*100:.2f}%", 'MDD': f"{(y_df_yr['Total']/y_df_yr['Total'].cummax()-1).min()*100:.2f}%"})
                prev_v = y_e
            st.table(pd.DataFrame(summary))
            st.line_chart(res_df['Total'])
