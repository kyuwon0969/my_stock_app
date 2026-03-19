import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 매니저", page_icon="🌿", layout="wide")

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
        # 가이드 표기 및 x값 계산을 위한 데이터
        df['p1_c'] = df['close'].shift(1)
        df['p2_c'] = df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        
        return df.loc[start_date:].dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

def run_simulation(df, initial_seed, num_slots):
    """Ivy 수익 저축 & BOXX 운용 & 슬롯별 상세 내역 추적 엔진"""
    if df is None or df.empty: return pd.DataFrame(), []
    
    cash, shares, used_slots, slot_cash, avg_price = float(initial_seed), 0.0, 0, 0.0, 0.0
    ivy_reserve = 0.0  
    boxx_rate = (1 + 0.053) ** (1/252) - 1 # 연 5.3% 일복리
    
    history = []
    slot_details = [] # 현재 보유 중인 슬롯들의 상세 리스트
    qqq_start_price = float(df['qqq_close'].iloc[0])
    
    for date, row in df.iterrows():
        # BOXX 이자 정산
        if ivy_reserve > 0:
            ivy_reserve *= (1 + boxx_rate)
            
        p1, p2 = row['p1_c'], row['p2_c']
        curr_close, qqq_curr_close, rsi_val = row['close'], row['qqq_close'], row['rsi']
        
        x_raw = (p1 + p2) * 1.01 / 1.99
        willow_x = np.ceil(x_raw * 100) / 100
        
        # 모드 및 타점 설정
        if rsi_val > 65: mode = "Ivy"; b_limit, s_limit = willow_x - 0.01, np.ceil((willow_x * 1.03) * 100) / 100
        elif rsi_val > 45: mode = "Willow"; b_limit, s_limit = willow_x - 0.01, willow_x
        elif rsi_val > 30: mode = "Lily"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x
        else: mode = "Tulip"; b_limit, s_limit = np.floor((willow_x * 0.975) * 100) / 100, willow_x

        sold_today = False
        # 매도 로직
        if shares > 0 and curr_close >= s_limit:
            sell_proceeds = shares * curr_close
            profit = sell_proceeds - (avg_price * shares)
            if mode == "Ivy" and profit > 0:
                cash += (avg_price * shares); ivy_reserve += profit        
            else:
                cash += sell_proceeds        
            shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
            slot_details = [] # 매도 시 상세내역 초기화
            sold_today = True
        
        # Tulip 진입 시 비상금 투입
        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        # 매수 로직
        if not sold_today and used_slots < num_slots and curr_close <= b_limit:
            if used_slots == 0: slot_cash = cash / num_slots
            buy_qty = slot_cash // curr_close
            if buy_qty > 0 and cash >= (buy_qty * curr_close):
                # 슬롯 상세 기록
                slot_details.append({
                    "슬롯": len(slot_details) + 1,
                    "날짜": date.strftime('%Y-%m-%d'),
                    "매수가": round(float(curr_close), 2),
                    "수량": int(buy_qty),
                    "금액": round(float(buy_qty * curr_close), 2)
                })
                avg_price = ((avg_price * shares) + (buy_qty * curr_close)) / (shares + buy_qty)
                shares += buy_qty
                cash -= (buy_qty * curr_close)
                used_slots += 1
        
        total_assets = cash + (shares * curr_close) + ivy_reserve
        history.append({
            'Date': date, 'Total': float(total_assets), 'Ivy_Reserve': float(ivy_reserve), 
            'Cash': float(cash), 'Shares': float(shares), 'Slots': int(used_slots), 
            'Avg_Price': float(avg_price), 'Slot_Cash': float(slot_cash),
            'QQQ_Hold': (initial_seed / qqq_start_price) * qqq_curr_close
        })
                    
    return pd.DataFrame(history).set_index('Date'), slot_details

# --- UI 레이아웃 ---
with st.sidebar:
    st.header("⚙️ 전략 및 운용 설정")
    target_ticker = st.selectbox("대상 종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"seasons_final_{target_ticker}"
    saved = localS.getItem(config_key) or {"op_start": "2024-01-01", "init_seed": 10000.0, "num_slots": 5}
    
    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(saved.get('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(saved['op_start']))
    init_seed = st.number_input("투자 원금 (USD)", value=float(saved['init_seed']), step=1000.0)
    
    if st.button("💾 설정값 저장"):
        localS.setItem(config_key, {"op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, "num_slots": num_slots})
        st.success("저장 완료!")
    if st.button("🔄 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 추적 & 가이드", "📊 과거 백테스트 리포트"])

with tab1:
    df_live = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if df_live is not None and not df_live.empty:
        hist_live, current_slots = run_simulation(df_live, init_seed, num_slots)
        cur, last = hist_live.iloc[-1], df_live.iloc[-1]
        
        # 1. 상단 현황판
        st.subheader(f"📊 {target_ticker} 운용 현황")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
        m2.metric("평균단가", f"${cur['Avg_Price']:.2f}")
        m3.metric("진행 회차", f"{int(cur['Slots'])} / {num_slots} 슬롯")
        m4.metric("현재 총 자산", f"${cur['Total']:,.2f}")
        
        st.info(f"🏦 현재 Ivy 비상금(BOXX) 저축액: **${cur['Ivy_Reserve']:,.2f}**")

        # 2. 보유 슬롯 상세 표 (요청 기능)
        if current_slots:
            st.markdown("#### 📝 현재 보유 슬롯 상세 내역")
            st.table(pd.DataFrame(current_slots))
        else:
            st.info("현재 보유 중인 물량이 없습니다.")

        # 3. 실전 주문 가이드 (데이터 포함)
        st.divider()
        rsi_now, p1, p2 = last['rsi'], last['p1_c'], last['p2_c']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        st.markdown(f"### 🎯 오늘의 실전 주문 가이드")
        st.write(f"🔍 **판단 근거:** QQQ RSI `{rsi_now:.2f}` | p1 `${p1:.2f}` | p2 `${p2:.2f}`")
        
        cl, cr = st.columns(2)
        with cl:
            st.success(f"#### 📥 {int(cur['Slots']) + 1}회차 매수 (LOC)")
            if int(cur['Slots']) < num_slots:
                target_slot_cash = cur['Cash'] / (num_slots - int(cur['Slots'])) if int(cur['Slots']) == 0 else cur['Slot_Cash']
                st.write(f"가격: `${x-0.01:.2f}` 이하 | 권장수량: `{int(target_slot_cash // (x-0.01))} 주`")
            else: st.write("✅ 풀매수 완료")
        with cr:
            # 모드에 따른 매도 타점 표시
            s_price = np.ceil((x * 1.03) * 100) / 100 if rsi_now > 65 else x
            st.error("#### 📤 전량 매도 (LOC)")
            if cur['Shares'] > 0:
                st.write(f"가격: `${s_price:.2f}` 이상 | 수량: `{int(cur['Shares'])} 주`")
            else: st.write("보유 물량 없음")
            
        st.line_chart(hist_live[['Total', 'QQQ_Hold']])

with tab2:
    st.header("📊 과거 백테스트 리포트")
    # 백테스트 탭에서도 동일한 시뮬레이션 엔진을 사용하여 일관성을 유지합니다.
    if st.button("🚀 13년 백테스트 실행"):
        df_back = get_processed_data(target_ticker, "2013-01-01")
        if df_back is not None:
            hist_back, _ = run_simulation(df_back, init_seed, num_slots)
            # (수익률, MDD 등 분석 지표 출력 로직 포함...)
            f_v = hist_back['Total'].iloc[-1]
            st.write(f"최종 자산: ${f_v:,.2f} | 수익률: {(f_v/init_seed-1)*100:.2f}%")
            st.line_chart(hist_back['Total'])
