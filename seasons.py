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

# --- 시뮬레이션 엔진 ---
def run_simulation(df, initial_seed, num_slots, pcr=0.7, start_limit_date=None, end_limit_date=None, pending_dep=0.0, manual_withdrawn=0.0):
    if df is None or df.empty: return pd.DataFrame(), [], []
    
    sim_df = df.dropna(subset=['rsi']).copy()
    
    if start_limit_date:
        sim_df = sim_df[sim_df.index.date >= start_limit_date]
    if end_limit_date:
        sim_df = sim_df[sim_df.index.date < end_limit_date]

    if sim_df.empty: return pd.DataFrame(), [], []

    cash = float(initial_seed) - manual_withdrawn
    shares, used_slots, slot_cash, avg_price = 0.0, 0, 0.0, 0.0
    ivy_reserve, cumulative_withdrawn = 0.0, 0.0 
    boxx_rate = (1 + 0.053) ** (1/252) - 1 
    
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
        
        if used_slots == 0 and internal_pending != 0:
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

# --- UI 레이아웃 및 저장 로직 ---
with st.sidebar:
    st.header("⚙️ 운용 설정")
    target_ticker = st.selectbox("종목 선택", ["SOXL", "USD"], index=0)
    config_key = f"v5_pro_final_{target_ticker}"
    
    q_params = st.query_params
    saved_ls = localS.getItem(config_key) or {}
    
    def get_setting(key, default):
        if key in q_params: return q_params[key]
        return saved_ls.get(key, default)

    num_slots = st.select_slider("매수 슬롯 분할 수", options=[3, 4, 5, 6], value=int(get_setting('num_slots', 5)))
    op_start = st.date_input("실제 운용 시작일", value=pd.to_datetime(get_setting('op_start', "2024-01-01")).date())
    init_seed = st.number_input("투자 원금 ($)", value=float(get_setting('init_seed', 10000.0)), step=1000.0)
    pcr_val = st.slider("PCR (재투자 비중)", 0.0, 1.0, float(get_setting('pcr', 0.7)), 0.05)
    
    st.divider()
    st.subheader("💰 수기 자금 관리")
    p_dep = st.number_input("추가 입금/출금액 ($)", value=float(get_setting('pending_dep', 0.0)), 
                          help="전량 매도 후 현금 상태일 때 주문금액에 적용됩니다. 출금 금액은 음수로 쓰면 됩니다. Cycle 도중에(전액 현금 상태가 아닐 때) 출금하지 않는 것을 추천합니다.")

    if st.button("💾 설정값 저장 및 강제 새로고침"):
        st.query_params.update({
            "num_slots": num_slots, "op_start": op_start.strftime('%Y-%m-%d'),
            "init_seed": init_seed, "pcr": pcr_val, "pending_dep": p_dep
        })
        localS.setItem(config_key, {
            "op_start": op_start.strftime('%Y-%m-%d'), "init_seed": init_seed, 
            "num_slots": num_slots, "pcr": pcr_val, "pending_dep": p_dep
        })
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🎯 실시간 현황 & 가이드", "📊 과거 데이터 기반 백테스트", "📖 Info (도움말)"])

with tab1:
    raw_df = get_processed_data(target_ticker, op_start.strftime('%Y-%m-%d'))
    if raw_df is not None:
        now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
        today_val = date.today()
        
        res_live, slots_live, _ = run_simulation(raw_df, init_seed, num_slots, pcr=pcr_val, start_limit_date=op_start, end_limit_date=today_val, pending_dep=p_dep)
        
        if not res_live.empty:
            valid_df = raw_df[raw_df.index.date < today_val]
            if valid_df.empty: valid_df = raw_df.iloc[:-1]
            
            latest_closed_row = valid_df.iloc[-1]
            prev_closed_row = valid_df.iloc[-2]
            p1_val, p2_val = latest_closed_row['close'], prev_closed_row['close']
            rsi_val = latest_closed_row['rsi_live']
            data_date = valid_df.index[-1].strftime('%Y-%m-%d')
            
            cur = res_live.iloc[-1]
            st.subheader(f"📊 {target_ticker} 현재 운용 현황")
            st.caption(f"🕒 최종 업데이트 (KST): {now_kst} | 📅 가이드 계산 기준일: {data_date}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
            c2.metric("평균 단가", f"${cur['Avg']:.2f}")
            c3.metric("채워진 슬롯", f"{int(cur['Slots'])} / {num_slots}")
            c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")
            
            st.info(f"🏦 Ivy 비상금: **${cur['Ivy']:,.2f}** | 💸 PCR 인출액: **${cur['Withdrawn']:,.2f}**")
            st.caption("ℹ️ Ivy 모드에서 낸 총 수익금은 'Ivy 비상금'탭에 표시됩니다. Ivy 비상금으로는 BOXX를 매수하시면 됩니다. BOXX를 매수하지 않고 현금으로 남겨두는 것도 괜찮습니다. 이 돈은 Tulip 모드에서 전량 사용되니, 따로 인출해서 쓰면 안 됩니다.")

            if slots_live:
                st.markdown("#### 📝 확정된 보유 슬롯 내역")
                st.table(pd.DataFrame(slots_live))

            st.divider()
            
            x = np.ceil(((p1_val + p2_val) * 1.01 / 1.99) * 100) / 100
            mode, color = ("Ivy", "red") if rsi_val > 65 else ("Willow", "orange") if rsi_val > 45 else ("Lily", "blue") if rsi_val > 30 else ("Tulip", "purple")
            
            tulip_msg = " (만약 Ivy 비상금으로 BOXX를 매수한 상태라면, BOXX를 현재 가격으로 전량 매도하세요.)" if mode == "Tulip" and cur['Slots'] == 0 else ""
            
            st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}]{tulip_msg})")
            st.write(f"🔍 **판단 근거:** QQQ RSI(전날) `{rsi_val:.2f}` | p1(전날) `${p1_val:.2f}` | p2(전전날) `${p2_val:.2f}` | 기준 x값 `${x:.2f}`")
            
            g1, g2 = st.columns(2)
            with g1:
                b_p = x - 0.01 if rsi_val > 45 else np.floor((x * 0.975)*100)/100
                st.error(f"#### 📥 {int(cur['Slots'])+1}회차 매수 (LOC)")
                if cur['Slots'] < num_slots:
                    t_cash = cur['Cash'] / (num_slots - cur['Slots'])
                    st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(t_cash // b_p)} 주`")
                else: st.write("✅ 매수 완료")
            with g2:
                s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                st.info(f"#### 📤 전량 매도 (LOC)")
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
            res_b, _, trades = run_simulation(bt_raw, bt_seed, num_slots, pcr=pcr_val, start_limit_date=bt_start, end_limit_date=bt_end + timedelta(days=1), pending_dep=0.0)
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
                m1.markdown(f"<p style='font-size: 1.25rem; color: gray; margin-top: -15px;'>({int(bt_seed * today_fx):,}원)</p>", unsafe_allow_html=True)
                m2.metric("최종 자산", f"${f_val:,.0f}")
                m2.markdown(f"<p style='font-size: 1.25rem; color: gray; margin-top: -15px;'>({int(f_val * today_fx):,}원)</p>", unsafe_allow_html=True)
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

with tab3:
    st.header("📖 사계절 전략 Pro 이용 가이드")
    
    # 카테고리 선택 (익스팬더 대신 가독성을 위해 셀렉트박스 활용)
    info_category = st.radio("궁금한 항목을 선택하세요", 
                             ["🎯 실시간 현황 및 가이드 설명", "📊 백테스트 용어 설명", "📥 LOC 주문 방법 (토스증권)", "💰 수기 자금 관리"],
                             horizontal=True)
    
    st.divider()

    if info_category == "🎯 실시간 현황 및 가이드 설명":
        st.subheader("1. 주요 수치 및 위젯 설명")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            * **총 수익률**: 원금 대비 현재 자산이 얼마나 늘었는지(또는 줄었는지)를 백분율로 보여줍니다.
            * **평균 단가**: 현재 보유 중인 주식들의 평균 매수 가격입니다.
            * **채워진 슬롯**: 전체 투자금을 몇 번에 나누어 살 것인지 중, 현재 몇 번째까지 매수했는지를 보여줍니다.
            * **현재 창출 가치**: 현금 + 주식 평가액 + 비상금 등을 모두 합친 나의 총 자산입니다.
            """)
        with col_info2:
            st.markdown("""
            * **QQQ RSI, p1, p2, x값(판단 근거)**: 매수/매도 주문 타점을 구할 때 사용하는 요소들입니다.
            * **Ivy 비상금**: 시장 상황이 좋을 때 챙겨두는 '보너스 수익금'입니다. 나중에 시장이 어려울 때 구원 투수로 사용됩니다.
            * **PCR 인출액**: 수익이 날 때마다 원금에 합치지 않고 따로 현금화하여 챙겨둔 금액입니다.
            """)
        
        st.divider()
        st.subheader("2. 오늘의 실전 가이드 활용법")
        st.info("매일 밤, 이 가이드를 보고 증권사 앱에서 **LOC 주문**을 예약하시면 됩니다.")
        st.markdown("""
        * **📥 매수 가이드 (빨간색 위젯)**: 
            - **타점**: 해당 금액 '이하'로 떨어지면 사겠다는 의미입니다.
            - **정량**: 가이드에 적힌 주수만큼 주문을 넣으시면 됩니다.
        * **📤 매도 가이드 (파란색 위젯)**: 
            - **타점**: 해당 금액 '이상'으로 오르면 전량 팔겠다는 의미입니다.
            - **수량**: 내가 가진 모든 주수를 입력하여 주문을 넣습니다.
        """)

    elif info_category == "📊 백테스트 용어 설명":
        st.subheader("과거 데이터 기반 백테스트란?")
        st.write("선택한 과거 기간 동안 이 전략을 그대로 실행했을 때 어떤 결과가 나왔을지 시뮬레이션하는 기능입니다.")
        
        st.markdown("""
        * **CAGR (연복리 수익률)**: 매년 평균적으로 자산이 몇 %씩 성장했는지를 나타냅니다.
        * **MDD (최대 낙폭)**: 전고점 대비 자산이 가장 많이 떨어졌을 때 몇 %나 하락했는지를 나타냅니다. (낮을수록 안전합니다.)
        * **Calmar (칼마 지수)**: CAGR을 MDD로 나눈 값입니다. 하락 위험 대비 수익 효율이 얼마나 좋은지 보여주는 지표입니다.
        * **그래프 설명**:
            - **Total (파란선)**: 본 전략을 사용했을 때의 자산 변화입니다.
            - **QQQ (오렌지선)**: 미국 지수인 **나스닥 100**을 추종하는 ETF입니다. 전략의 성능을 시장 지수와 비교하기 위해 표시됩니다.
        """)

    elif info_category == "📥 LOC 주문 방법 (토스증권)":
        st.subheader("토스증권 LOC 주문 단계별 가이드")
        st.write("LOC(Limit On Close) 주문은 장 마감 가격이 내가 정한 가격보다 유리할 때만 체결되는 주문 방식입니다.")
        
        st.markdown("1. 주문 종류를 **'LOC'**로 변경합니다.")
        st.image("toss1.jpg", width=350)
        
        st.markdown("2. 가격에 가이드의 **'타점'** 금액을 입력합니다.")
        st.image("toss2.jpg", width=350)
        
        st.markdown("3. 수량에 가이드의 **'정량'** 주수를 입력하고 '매수'를 누릅니다.")
        st.image("toss3.jpg", width=350)

    elif info_category == "💰 수기 자금 관리":
        st.subheader("입출금 및 자금 관리 주의사항")
        st.markdown("""
        * **추가 입금/출금액**: 계좌에 돈을 더 넣거나 빼고 싶을 때 사용합니다. 
        * **입금**: 양수(예: 1000)를 입력하세요.
        * **출금**: 음수(예: -1000)를 입력하세요.
        * **주의**: 주식을 하나라도 보유 중일 때는 계산이 꼬일 수 있으니, 모든 주식을 다 팔고 **'전액 현금'** 상태일 때만 적용하는 것을 강력 추천합니다.
        """)
