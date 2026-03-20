import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from datetime import datetime, date, timedelta
import pytz # 시간대 처리를 위해 추가
from streamlit_local_storage import LocalStorage

# 1. 페이지 설정
st.set_page_config(page_title="사계절 전략 Pro", page_icon="🌿", layout="wide")
localS = LocalStorage()

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# --- 데이터 엔진 (기존 유지) ---
@st.cache_data(ttl=300)
def get_processed_data(ticker, start_date):
    try:
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
        
        df['p1_c'], df['p2_c'] = df['close'].shift(1), df['close'].shift(2)
        
        delta = qqq_close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        df['rsi_live'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan)))))
        
        return df 
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None

# --- 시뮬레이션 엔진 (입출금 로직 반영 - 기존 유지) ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None, pending_dep=0.0, manual_withdrawn=0.0):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.dropna(subset=['rsi']).copy()
    
    if start_limit_date:
        sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date:
        sim_df = sim_df[sim_df.index.date < end_limit_date]

    if sim_df.empty: return pd.DataFrame(), [], []

    # 초기 현금에서 인출 금액 반영
    cash = float(initial_seed) - manual_withdrawn
    shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
    ivy_reserve, cumulative_withdrawn = 0.0, 0.0 
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
    # 추가 입금 보류 로직용 변수
    internal_pending = float(pending_dep)
    
    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
            
        p1, p2, curr_c, rsi_v = row['p1_c'], row['p2_c'], row['close'], row['rsi']
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
                withdrawn_profit = profit * (1 - pcr)
                if mode == "Ivy": ivy_reserve += comp_profit
                else: cash += comp_profit
                cumulative_withdrawn += withdrawn_profit
            else:
                cash += profit
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []
            sold = True
        
        # 추가 금액 로직: 차기 사이클 시작 시 보류된 입금액 자동 투입
        if used_slots == 0 and internal_pending > 0:
            cash += internal_pending
            internal_pending = 0.0

        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
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
        
        total_assets = cash + (shares * curr_c) + ivy_reserve + cumulative_withdrawn
        history.append({
            'Date': date_idx, 'Total': total_assets, 'Cash': cash, 'Shares': shares, 
            'Slots': used_slots, 'Avg': avg_price, 'Withdrawn': cumulative_withdrawn,
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close'], 'Ivy': ivy_reserve
        })
    return pd.DataFrame(history).set_index('Date'), slot_details, trade_profits

# --- UI 레이아웃 및 저장 로직 (새로고침 대응 강화) ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"v5_pro_final_{target_ticker}"
    
    # [핵심] URL 쿼리 파라미터에서 데이터 읽기
    q_params = st.query_params
    
    # 저장된 데이터 로드 (URL -> LocalStorage -> Default 순)
    saved_ls = localS.getItem(config_key) or {}
    
    def get_setting(key, default):
        # 1. URL에 값이 있는지 확인 (새로고침 대응)
        if key in q_params:
            return q_params[key]
        # 2. 로컬 스토리지 확인
        return saved_ls.get(key, default)

    # 위젯 초기값 설정
    def_slots = int(get_setting('num_slots', 5))
    def_start = get_setting('op_start', "2024-01-01")
    def_seed = float(get_setting('init_seed', 10000.0))
    def_pcr = float(get_setting('pcr', 0.7))
    def_pending = float(get_setting('pending_dep', 0.0))
    def_manual_with = float(get_setting('manual_withdrawn', 0.0))

    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=def_slots)
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(def_start).date())
    init_seed = st.number_input("투자 원금 ($)", value=def_seed, step=1000.0)
    pcr_val = st.slider("PCR (재투자 비중)", 0.0, 1.0, def_pcr, 0.05)
    
    st.divider()
    st.subheader("💰 수기 자금 관리")
    p_dep = st.number_input("보류 중인 추가 입금액 ($)", value=def_pending, help="전량 매도 후 현금 상태일 때 투입됩니다.")
    m_with = def_manual_with # 인출 금액

    if st.button("💾 설정값 저장 및 강제 새로고침"):
        # URL 쿼리 업데이트
        st.query_params.update({
            "num_slots": num_slots,
            "op_start": op_start.strftime('%Y-%m-%d'),
            "init_seed": init_seed,
            "pcr": pcr_val,
            "pending_dep": p_dep,
            "manual_withdrawn": m_with
        })
        # 로컬 스토리지 백업
        localS.setItem(config_key, {
            "op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, 
            "num_slots": num_slots, "pcr": pcr_val, "pending_dep": p_dep, "manual_withdrawn": m_with
        })
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트"])

with tab1:
    # 1. 데이터 가져오기 (충분한 과거 데이터를 위해 op_start보다 여유 있게 가져옴)
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    
    if raw_df is not None and len(raw_df) >= 2:
        now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
        today_val = date.today()
        
        # 시뮬레이션 실행 (운용 시작일부터 오늘 직전까지의 기록)
        res_live, slots_live, _ = run_simulation(
            raw_df, init_seed, num_slots, pcr=pcr_val, 
            start_limit_date=op_start, end_limit_date=today_val, 
            pending_dep=p_dep, manual_withdrawn=m_with
        )
        
        if not res_live.empty:
            # --- [수정 구간] 최신 데이터 추출 로직 ---
            # raw_df의 마지막 행이 '어제' 혹은 '가장 최근 장마감' 데이터입니다.
            latest_data = raw_df.iloc[-1]   # 전날 (Last Trading Day)
            prev_data = raw_df.iloc[-2]     # 전전날 (2nd Last Trading Day)
            
            p1_val = float(latest_data['close'])      # 전날 SOXL 종가
            p2_val = float(prev_data['close'])       # 전전날 SOXL 종가
            rsi_val = float(latest_data['rsi_live'])  # 전날 QQQ RSI (rsi_live가 당일 기준이므로)
            
            # 기준 날짜 확인용
            data_date = raw_df.index[-1].strftime('%Y-%m-%d')
            prev_date = raw_df.index[-2].strftime('%Y-%m-%d')
            # ---------------------------------------

            cur = res_live.iloc[-1]
            st.subheader(f"📊 {target_ticker} 현재 운용 현황")
            st.caption(f"🕒 업데이트: {now_kst} | 📅 데이터 기준: {data_date}(p1), {prev_date}(p2)")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("평균 단가", f"${cur['Avg']:.2f}")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}")
            c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")
            
            st.info(f"🏦 Ivy 비상금: **${cur['Ivy']:,.2f}** | 💸 PCR 인출액: **${cur['Withdrawn']:,.2f}**")
            
            # 수기 인출 로직 (기존 유지)
            if int(cur['Slots']) == 0:
                st.success("✅ 현재 모든 슬롯이 비어 있어 '인출'이 가능합니다.")
                new_withdraw = st.number_input("수기 인출 금액 입력 ($)", value=m_with, step=100.0)
                if new_withdraw != m_with:
                    st.query_params["manual_withdrawn"] = new_withdraw
                    localS.setItem(config_key, {
                        "op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, 
                        "num_slots": num_slots, "pcr": pcr_val, "pending_dep": p_dep, "manual_withdrawn": new_withdraw
                    })
                    if st.button("인출 확정 및 반영"): st.rerun()
            else:
                st.error("⚠️ 주식 보유 중에는 '인출' 기능을 사용할 수 없습니다.")

            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))

            st.divider()
            
            # --- [가이드 계산] 수정된 p1, p2, rsi 적용 ---
            # 계산식: x = ceil((p1 + p2) * 1.01 / 1.99)
            x = np.ceil(((p1_val + p2_val) * 1.01 / 1.99) * 100) / 100
            
            # 모드 결정
            if rsi_val > 65: mode, color = "Ivy", "red"
            elif rsi_val > 45: mode, color = "Willow", "orange"
            elif rsi_val > 30: mode, color = "Lily", "blue"
            else: mode, color = "Tulip", "purple"
            
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
            st.write(f"🔍 **판단 근거:** QQQ RSI(전날) `{rsi_val:.2f}` | p1(전날) `${p1_val:.2f}` | p2(전전날) `${p2_val:.2f}` | 기준 x값 `${x:.2f}`")
            
            g1, g2 = st.columns(2)
            with g1:
                # 매수 타점: RSI > 45 이면 x-0.01, 아니면 x * 0.975
                b_p = x - 0.01 if rsi_val > 45 else np.floor((x * 0.975)*100)/100
                st.success(f"#### 📥 {int(cur['Slots'])+1}회차 매수 (LOC)")
                if cur['Slots'] < num_slots:
                    t_cash = cur['Cash'] / (num_slots - cur['Slots'])
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(t_cash // b_p)} 주`")
                else: st.write("✅ 매수 완료 (모든 슬롯 소진)")
            with g2:
                # 매도 타점: RSI > 65 이면 x * 1.03, 아니면 x
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                st.error("#### 📤 전량 매도 (LOC)")
                if cur['Shares'] > 0:
                    st.write(f"**타점:** `${s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                else: st.write("보유 물량 없음")
            
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
            res_b, _, trades = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1), pending_dep=0.0, manual_withdrawn=0.0)
            if not res_b.empty:
                f_val, withdrawn = res_b['Total'].iloc[-1], res_b['Withdrawn'].iloc[-1]
                cagr = ((f_val / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                peak = res_b['Total'].cummax()
                mdd = (res_b['Total'] / peak - 1).min() * 100
                
                try:
                    fx_data = yf.download("USDKRW=X", period="1d", progress=False)
                    today_fx = float(fx_data['Close'].iloc[-1])
                except: today_fx = 1350.0
                
                st.divider()
                st.subheader("🏆 백테스트 종합 결과")
                st.caption(f"ℹ️ 적용된 오늘 환율: {today_fx:,.2f}원")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("초기 자산", f"${bt_seed:,.0f}")
                m1.markdown(f"<p style='font-size: 0.85rem; color: gray; margin-top: -15px;'>({int(bt_seed * today_fx):,}원)</p>", unsafe_allow_html=True)
                m2.metric("최종 자산", f"${f_val:,.0f}")
                m2.markdown(f"<p style='font-size: 0.85rem; color: gray; margin-top: -15px;'>({int(f_val * today_fx):,}원)</p>", unsafe_allow_html=True)
                m3.metric("CAGR (연복리)", f"{cagr:.2f}%")
                
                m4, m5, m6 = st.columns(3)
                m4.metric("MDD", f"{mdd:.2f}%"); m5.metric("Calmar", f"{cagr/abs(mdd):.2f}"); m6.metric("총 인출 현금", f"${withdrawn:,.0f}")

                st.line_chart(res_b[['Total', 'QQQ']])
                res_b['year'] = res_b.index.year
                y_stats = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    y_stats.append({"연도": yr, "수익률": f"{(y_df['Total'].iloc[-1]/y_df['Total'].iloc[0]-1)*100:.1f}%", "MDD": f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.1f}%", "연간 인출": f"${(y_df['Withdrawn'].iloc[-1] - y_df['Withdrawn'].iloc[0]):,.0f}"})
                st.table(pd.DataFrame(y_stats))
