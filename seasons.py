import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 통합 매니저", page_icon="🌿", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 ---
@st.cache_data(ttl=600)
def get_processed_data(ticker, start_date):
    """지정한 기간의 데이터를 가져오고 지표 계산"""
    try:
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        # 최신 데이터 확보를 위해 end를 지정하지 않음
        data = yf.download([ticker, "QQQ"], start=fetch_start, progress=False)
        
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].dropna()
            qqq_close = data['Close']['QQQ'].dropna()
        else:
            target_close = data[ticker].dropna()
            qqq_close = data['QQQ'].dropna()
        
        df = pd.DataFrame(index=target_close.index)
        df['close'] = target_close
        df['qqq_close'] = qqq_close
        df['prev_close'] = df['close'].shift(1)
        df['prev_close2'] = df['close'].shift(2)
        
        # Wilder's RSI (QQQ 기준)
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"데이터 수집 중 오류: {e}")
        return None

def run_simulation(df, initial_seed):
    """사계절 전략 시뮬레이션 엔진 (고정 슬롯 매수)"""
    cash, shares, used_slots, slot_cash, avg_price = initial_seed, 0, 0, 0, 0
    history = []
    
    if df.empty: return pd.DataFrame()
    qqq_start_price = df['qqq_close'].iloc[0]
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, qqq_curr_close, rsi_val = row['prev_close'], row['prev_close2'], row['close'], row['qqq_close'], row['rsi']
        
        x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        if shares > 0 and curr_close >= s_limit:
            cash += (shares * curr_close)
            shares, used_slots, slot_cash, avg_price = 0, 0, 0, 0
            sold_today = True
        
        if not sold_today and used_slots < 5 and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / 5
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        total_assets = cash + (shares * curr_close)
        qqq_hold_val = (initial_seed / qqq_start_price) * qqq_curr_close
        
        history.append({
            'Date': date, 'Total': total_assets, 'QQQ_Hold': qqq_hold_val,
            'QQQ_Price': qqq_curr_close, 'Slots': used_slots, 'Avg_Price': avg_price,
            'Cash': cash, 'Shares': shares, 'Slot_Cash': slot_cash
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
st.title("🌿 사계절 전략 통합 매니저")

with st.sidebar:
    st.header("⚙️ 기본 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD", "QLD"], index=0)
    
    st.divider()
    config_key = f"seasons_config_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0}
    
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    
    if st.button("💾 이 종목 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed})
        st.success("설정 저장 완료!")

    if st.button("🔄 데이터 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

# --- TAB 1: 실시간 추적 ---
with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed)
        cur = hist_live.iloc[-1]
        last_data = df_live.iloc[-1]
        
        total_ret = (cur['Total'] / init_seed - 1) * 100
        qqq_ret = (cur['QQQ_Hold'] / init_seed - 1) * 100

        st.subheader(f"📊 {target_ticker} 운용 현황 (데이터: {df_live.index[-1].strftime('%Y-%m-%d')})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전략 수익률", f"{total_ret:+.2f}%", f"QQQ 대비 {total_ret-qqq_ret:+.2f}%")
        m2.metric("평균 단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / 5 슬롯")
        m4.metric("현재 총 자산", f"${cur['Total']:,.2f}")

        st.divider()
        rsi_now = last_data['rsi']
        p1, p2 = last_data['prev_close'], last_data['prev_close2']
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        if rsi_now > 65: mode, color, b_l, s_l = "Ivy", "red", willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_now > 45: mode, color, b_l, s_l = "Willow", "orange", willow_x - 0.01, willow_x
        elif rsi_now > 30: mode, color, b_l, s_l = "Lily", "blue", np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode, color, b_l, s_l = "Tulip", "purple", np.floor((willow_x * 0.975) * 100) / 100, willow_x

        st.markdown(f"### 🎯 오늘의 실전 주문 가이드 (모드: :{color}[{mode}])")
        cl, cr = st.columns(2)
        
        with cl:
            st.success(f"#### 📥 {int(cur['Slots']) + 1}회차 매수 예약 (LOC)")
            if cur['Slots'] < 5:
                # 다음 매수 수량 계산
                # 만약 현재 슬롯이 0이라면 현재 현금의 1/5, 아니면 이미 정해진 slot_cash 사용
                target_slot_cash = cur['Cash'] / 5 if cur['Slots'] == 0 else cur['Slot_Cash']
                buy_qty = int(target_slot_cash // b_l)
                
                st.write(f"**매수 지정가:** `${b_l:.2f}` 이하")
                st.write(f"**매수 수량:** `{buy_qty} 주` (권장)")
                st.caption(f"기준: 1슬롯 예산 ${target_slot_cash:,.2f}")
            else:
                st.write("✅ 모든 슬롯 매수 완료. 매도 시점을 기다리세요.")
        
        with cr:
            st.error("#### 📤 전량 매도 예약 (LOC)")
            if cur['Shares'] > 0:
                st.write(f"**매도 지정가:** `${s_l:.2f}` 이상")
                st.write(f"**매도 수량:** `{int(cur['Shares'])} 주` (보유 전량)")
                st.write(f"**목표 수익률:** `{(s_l/cur['Avg_Price']-1)*100:+.2f}%` (평단 대비)")
            else:
                st.write("보유 물량이 없습니다.")

        st.divider()
        st.subheader("📈 자산 흐름 비교 (파랑: 전략 / 주황: QQQ)")
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])
    else:
        st.warning("데이터를 가져오는 중입니다. 잠시만 기다려주세요.")

# --- TAB 2: 백테스트 (생략 없이 유지) ---
with tab2:
    st.header("🔍 초장기 과거 성과 분석")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("테스트 시작일", value=datetime(2013, 1, 1), min_value=datetime(2013, 1, 1))
    with c2: e_date = st.date_input("테스트 종료일", value=datetime.now())
    with c3: s_seed = st.number_input("테스트 시드", value=10000.0, step=1000.0, key="backtest_seed")

    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, s_date.strftime('%Y-%m-%d'))
        # 백테스트는 종료일 슬라이싱 필요
        df_back = df_back.loc[:e_date.strftime('%Y-%m-%d')] if df_back is not None else None
        
        if df_back is not None and not df_back.empty:
            res_back = run_simulation(df_back, s_seed)
            final_v = res_back['Total'].iloc[-1]
            final_qqq = res_back['QQQ_Hold'].iloc[-1]
            total_ret, qqq_ret = (final_v / s_seed - 1) * 100, (final_qqq / s_seed - 1) * 100
            days = (res_back.index[-1] - res_back.index[0]).days
            cagr = ((final_v / s_seed) ** (365.25 / (days if days > 0 else 1)) - 1) * 100
            mdd = (res_back['Total'] / res_back['Total'].cummax() - 1).min() * 100

            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("최종 자산", f"${final_v:,.2f}")
            r2.metric("총 수익률", f"{total_ret:,.2f}%", f"QQQ 대비 {total_ret-qqq_ret:+.2f}%")
            r3.metric("CAGR (연복리)", f"{cagr:.2f}%")
            r4.metric("MDD (최대낙폭)", f"{mdd:.2f}%")
            st.line_chart(res_back[['Total', 'QQQ_Hold']])
