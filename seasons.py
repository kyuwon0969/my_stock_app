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
            
        # [수정 로직] N+1번째 예비 슬롯까지 허용
        if not sold and used_slots < (num_slots + 1) and curr_c <= b_l:
            # 기본 슬롯 배분
            if used_slots == 0: 
                slot_cash = cash / num_slots
            
            # 매수 자금 결정: N회차까지는 정량, N+1회차(예비)는 잔여 현금 전량
            current_order_cash = cash if used_slots >= num_slots else slot_cash
            
            buy_qty = current_order_cash // b_l 
            actual_cost = buy_qty * curr_c
            if buy_qty > 0 and cash >= actual_cost:
                slot_label = f"예비" if used_slots >= num_slots else used_slots + 1
                slot_details.append({
                    "슬롯": slot_label, "날짜": date_idx.strftime('%Y-%m-%d'),
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
    target_ticker = "SOXL"
    st.info(f"📍 종목: **{target_ticker} (고정)**")
    
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
                          help="전량 매도 후 현금 상태일 때 주문금액에 적용됩니다.")

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
            
            if len(valid_df) >= 2:
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
                
                # 슬롯 표시 로직 (예비 슬롯 포함)
                slot_display = f"{int(cur['Slots'])} / {num_slots}" if cur['Slots'] <= num_slots else f"{num_slots} + 예비"
                c3.metric("채워진 슬롯", slot_display)
                c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")
                st.info(f"🏦 Ivy 비상금: **${cur['Ivy']:,.2f}** | 💸 PCR 인출액: **${cur['Withdrawn']:,.2f}**")
                
                if slots_live:
                    st.markdown("#### 📝 확정된 보유 슬롯 내역")
                    st.table(pd.DataFrame(slots_live))
                st.divider()
                
                x = np.ceil(((p1_val + p2_val) * 1.01 / 1.99) * 100) / 100
                mode, color = ("Ivy", "red") if rsi_val > 65 else ("Willow", "orange") if rsi_val > 45 else ("Lily", "blue") if rsi_val > 30 else ("Tulip", "purple")
                tulip_msg = " (만약 Ivy 비상금으로 BOXX를 매수한 상태라면, BOXX를 현재 가격으로 전량 매도하세요.)" if mode == "Tulip" and cur['Slots'] == 0 else ""
                st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}]{tulip_msg})")
                
                g1, g2 = st.columns(2)
                with g1:
                    b_p = x - 0.01 if rsi_val > 45 else np.floor((x * 0.975)*100)/100
                    
                    # [가이드 로직 수정] 예비 슬롯 주문 안내 추가
                    if cur['Slots'] < num_slots:
                        st.error(f"#### {int(cur['Slots'])+1}회차 정규 매수 (LOC)")
                        try:
                            zero_slots_df = res_live[res_live['Slots'] == 0]
                            base_capital = zero_slots_df.iloc[-1]['Cash'] if not zero_slots_df.empty else init_seed
                            fund = base_capital + p_dep
                            if mode == "Tulip" and cur['Slots'] == 0: fund += cur['Ivy']
                            order_cash = fund / num_slots
                            st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(order_cash // b_p)} 주`")
                        except: st.write("⚠️ 계산 오류")
                    elif cur['Slots'] == num_slots:
                        st.warning(f"#### 🔥 예비 슬롯 추가 매수 (LOC)")
                        # 예비 슬롯은 현재 남은 모든 현금 사용 (Ivy 제외)
                        order_cash = cur['Cash']
                        st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(order_cash // b_p)} 주`")
                        st.caption("💡 기존 분할 완료 후 남은 현금을 모두 투입하는 예비 단계입니다.")
                    else:
                        st.write("✅ 매수 완료 (최대 슬롯 도달)")

                with g2:
                    s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                    st.info(f"#### 📤 전량 매도 (LOC)")
                    if cur['Shares'] > 0:
                        st.write(f"**타점:** `${s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                    else: st.write("보유 없음")
                st.line_chart(res_live[['Total', 'QQQ']])

# --- 백테스트 및 도움말 (기존 동일) ---
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
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("초기 자산", f"${bt_seed:,.0f}")
                m2.metric("최종 자산", f"${f_val:,.0f}")
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
    st.markdown("""### 🆕 예비 슬롯(+1) 로직 안내
기존에는 설정한 분할 수가 끝나면 추가 매수를 하지 않았으나, 본 버전은 **'예비 슬롯'** 기능을 지원합니다.
1. **정규 슬롯 (1~N)**: 설정한 원금을 N등분하여 기계적으로 매수합니다.
2. **예비 슬롯 (N+1)**: 정규 매수 완료 후에도 가격이 타점 이하일 경우, 계좌에 남은 **모든 현금을 투입**하여 단가를 한 번 더 낮춥니다.
3. **효과**: 현금 놀리는 구간을 최소화하고, 깊은 하락장에서 더 강력하게 대응합니다.""")
