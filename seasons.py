import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 통합 매니저", page_icon="🌿", layout="wide")

localS = LocalStorage()

# --- 데이터 엔진 (에러 방지 및 로딩 최적화) ---
@st.cache_data(ttl=600)
def get_processed_data(ticker, start_date):
    """데이터 수집 및 지표 계산 (무한 로딩 방지 로직 추가)"""
    try:
        # RSI 예열을 위해 시작일 6개월 전부터 수집
        fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
        
        # 데이터 수집 (안정성을 위해 최근 데이터 위주로 먼저 확인)
        # 종목과 QQQ를 개별적으로 받아서 합치는 방식이 더 안정적일 때가 있습니다.
        tickers = [ticker, "QQQ"]
        data = yf.download(tickers, start=fetch_start, progress=False)
        
        if data.empty or len(data) < 20:
            return None
        
        # 컬럼 구조 정리 (Single/Multi Index 대응)
        if isinstance(data.columns, pd.MultiIndex):
            target_close = data['Close'][ticker].ffill()
            qqq_close = data['Close']['QQQ'].ffill()
        else:
            # 데이터가 하나만 올 경우를 대비
            target_close = data['Close'].ffill() if ticker in data.columns else data.iloc[:, 0].ffill()
            qqq_close = data['Close'].ffill() # 실제로는 위 조건문에서 걸러짐
        
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
        st.error(f"⚠️ 데이터 수집 중 오류 발생: {e}")
        return None

def run_simulation(df, initial_seed):
    """사계절 전략 시뮬레이션 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = initial_seed, 0, 0, 0, 0
    history = []
    qqq_start_price = df['qqq_close'].iloc[0]
    
    for date, row in df.iterrows():
        p_prev1, p_prev2, curr_close, qqq_curr_close, rsi_val = row['prev_close'], row['prev_close2'], row['close'], row['qqq_close'], row['rsi']
        
        # 타점 연산
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
            'Cash': cash, 'Shares': shares, 'Slot_Cash': slot_cash, 'Avg_Price': avg_price
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
    
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed})
        st.success("저장되었습니다!")

    if st.button("🔄 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

# --- TAB 1: 실시간 추적 ---
with tab1:
    # 빵 굽는 애니메이션 대신 진행 상황을 알 수 있도록 문구 수정
    with st.status("📡 시장 데이터를 연결 중입니다...", expanded=True) as status:
        df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
        if df_live is not None:
            status.update(label="✅ 데이터 수집 완료!", state="complete", expanded=False)
        else:
            status.update(label="❌ 데이터 수집 실패", state="error")

    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed)
        cur = hist_live.iloc[-1]
        last_data = df_live.iloc[-1]
        
        total_ret = (cur['Total'] / init_seed - 1) * 100
        qqq_ret = (cur['QQQ_Hold'] / init_seed - 1) * 100

        st.subheader(f"📊 {target_ticker} 운용 현황 ({df_live.index[-1].strftime('%Y-%m-%d')})")
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
            st.success(f"#### 📥 {int(cur['Slots']) + 1}회차 매수 (LOC)")
            if cur['Slots'] < 5:
                target_slot_cash = cur['Cash'] / 5 if cur['Slots'] == 0 else cur['Slot_Cash']
                buy_qty = int(target_slot_cash // b_l)
                st.write(f"**매수 가격:** `${b_l:.2f}` 이하")
                st.write(f"**매수 수량:** `{buy_qty} 주` 권장")
                st.caption(f"기준 RSI: {rsi_now:.2f} | p1: ${p1:.2f} | p2: ${p2:.2f}")
            else: st.write("✅ 모든 슬롯 완료")
        with cr:
            st.error("#### 📤 전량 매도 (LOC)")
            if cur['Shares'] > 0:
                st.write(f"**매도 가격:** `${s_l:.2f}` 이상")
                st.write(f"**매도 수량:** `{int(cur['Shares'])} 주` (전량)")
            else: st.write("보유 물량 없음")

        st.divider()
        st.subheader("📈 자산 흐름 비교 (파랑: 전략 / 주황: QQQ)")
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])
    else:
        st.warning("📡 데이터를 가져오지 못했습니다. 잠시 후 [강제 새로고침]을 눌러주세요.")

# --- TAB 2: 백테스트 (로직 동일) ---
with tab2:
    st.header("🔍 초장기 과거 성과 분석")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("테스트 시작일", value=datetime(2013, 1, 1), min_value=datetime(2013, 1, 1))
    with c2: e_date = st.date_input("테스트 종료일", value=datetime.now())
    with col3: s_seed = st.number_input("테스트 시드", value=10000.0, step=1000.0, key="bt_seed")

    if st.button("🚀 백테스트 실행"):
        df_back = get_processed_data(target_ticker, s_date.strftime('%Y-%m-%d'))
        if df_back is not None:
            df_back = df_back.loc[:e_date.strftime('%Y-%m-%d')]
            res_back = run_simulation(df_back, s_seed)
            if not res_back.empty:
                final_v = res_back['Total'].iloc[-1]
                st.metric("최종 자산", f"${final_v:,.2f}")
                st.line_chart(res_back[['Total', 'QQQ_Hold']])
