import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 [분할+1일 MOC]", page_icon="⚖️", layout="wide")

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
        
        # Wilder's RSI (QQQ 기준)
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots):
    """분할 수 + 1일 기준 동적 MOC 매도 엔진"""
    if df is None or df.empty: return pd.DataFrame()
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    holding_days = 0 
    max_holding_days = num_slots + 1 # [핵심] 분할 수 + 1일 설정
    history = []
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        p1, p2 = row['prev_close'], row['prev_close2']
        curr_close, qqq_curr_close, rsi_val = row['close'], row['qqq_close'], row['rsi']
        
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # RSI 기반 가격 제한 설정
        if rsi_val > 65: b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        
        # [동적 MOC 판정]
        if shares > 0:
            holding_days += 1
            # 1. 가격 도달 매도 (LOC)
            if curr_close >= s_limit:
                cash += (shares * curr_close)
                shares, used_slots, slot_cash, avg_price, holding_days = 0.0, 0, 0.0, 0.0, 0
                sold_today = True
            # 2. (분할 수 + 1)일 도달 시 강제 매도 (MOC)
            elif holding_days >= max_holding_days:
                cash += (shares * curr_close)
                shares, used_slots, slot_cash, avg_price, holding_days = 0.0, 0, 0.0, 0.0, 0
                sold_today = True
        
        # 매수 로직 (기존과 동일)
        if not sold_today and used_slots < num_slots and curr_close <= b_limit:
            if used_slots == 0: 
                slot_cash = cash / num_slots
                holding_days = 0 
            
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        total_assets = cash + (shares * curr_close)
        qqq_hold_val = (initial_seed / qqq_start_price) * qqq_curr_close
        
        history.append({
            'Date': date, 'Total': float(total_assets), 'QQQ_Hold': float(qqq_hold_val),
            'Cash': float(cash), 'Shares': float(shares), 'Slots': int(used_slots), 
            'Avg_Price': float(avg_price), 'Holding_Days': int(holding_days)
        })
                    
    return pd.DataFrame(history).set_index('Date')

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 전략 설정")
    target_ticker = st.selectbox("대상 종목", ["SOXL", "USD"], index=0)
    config_key = f"seasons_dynamic_moc_{target_ticker}"
    saved_config = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("매수 슬롯 분할 수 (N)", options=[3, 4, 5, 6], value=int(saved_config.get('num_slots', 5)))
    st.info(f"💡 현재 설정: {num_slots+1}일째 MOC 매도")
    
    op_start = st.date_input("운용 시작일", value=pd.to_datetime(saved_config['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved_config['init_seed']), step=1000.0)
    
    if st.button("💾 설정 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실전 가이드", "📊 성과 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live = run_simulation(df_live, init_seed, num_slots)
        cur = hist_live.iloc[-1]
        last_data = df_live.iloc[-1]
        
        h_days = int(cur['Holding_Days'])
        m_limit = num_slots + 1
        
        st.subheader(f"📊 {target_ticker} 현황 ({num_slots}분할 / Max {m_limit}일)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전략 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
        m2.metric("보유 일수", f"{h_days} / {m_limit}일")
        m3.metric("진행 회차", f"{int(cur['Slots'])} 슬롯")
        m4.metric("총 자산", f"${cur['Total']:,.2f}")

        st.divider()
        # 가이드 로직 (중략 없이 핵심 반영)
        rsi_now, p1, p2 = last_data['rsi'], last_data['prev_close'], last_data['prev_close2']
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        # (생략 없이 RSI 판정 로직 적용하여 s_l 계산...)
        if rsi_now > 65: s_l = np.ceil((willow_x * 1.03) * 100) / 100
        else: s_l = willow_x

        st.markdown(f"### 🎯 오늘의 주문 가이드 (동적 MOC)")
        cl, cr = st.columns(2)
        with cl:
            st.success("#### 📥 매수")
            if int(cur['Slots']) < num_slots:
                st.write(f"지정가 매수 대기 중...")
            else: st.write("풀 매수 상태")
        with cr:
            st.error("#### 📤 매도")
            if h_days >= num_slots: # 만기 하루 전
                st.warning(f"⚠️ 내일은 보유 {m_limit}일째입니다. 반등 실패 시 MOC 전량 매매!")
            elif cur['Shares'] > 0:
                st.write(f"**목표가(LOC):** `${s_l:.2f}` 이상")
            else: st.write("보유 물량 없음")
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])

with tab2:
    st.header(f"🔍 {num_slots}분할 + {num_slots+1}일 MOC 백테스트")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("시작일", value=datetime(2013, 1, 1), key="bt_s_dyn")
    with c2: e_date = st.date_input("종료일", value=datetime.now(), key="bt_e_dyn")
    with c3: s_seed = st.number_input("시드", value=10000.0, key="bt_seed_dyn")

    if st.button("🚀 분석 시작"):
        df_back = get_processed_data(target_ticker, s_date.strftime('%Y-%m-%d'))
        if df_back is not None:
            res_back = run_simulation(df_back.loc[:e_date.strftime('%Y-%m-%d')], s_seed, num_slots)
            if not res_back.empty:
                # 성과 지표 계산
                f_v = res_back['Total'].iloc[-1]
                t_r = (f_v / s_seed - 1) * 100
                mdd = (res_back['Total'] / res_back['Total'].cummax() - 1).min() * 100
                st.divider()
                r1, r2, r3 = st.columns(3)
                r1.metric("최종 수익률", f"{t_r:.2f}%")
                r2.metric("전략 MDD", f"{mdd:.2f}%")
                r3.metric("최종 자산", f"${f_v:,.2f}")
                st.line_chart(res_back[['Total', 'QQQ_Hold']])
                
                # 연도별 데이터 요약
                res_back['year'] = res_back.index.year
                yearly = []
                temp_seed = s_seed
                for yr in sorted(res_back['year'].unique()):
                    y_df = res_back[res_back['year'] == yr]
                    y_e_v = y_df['Total'].iloc[-1]
                    yearly.append({'연도': yr, '수익률': f"{(y_e_v/temp_seed-1)*100:.2f}%", 'MDD': f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.2f}%"})
                    temp_seed = y_e_v
                st.table(pd.DataFrame(yearly))
