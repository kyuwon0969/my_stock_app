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
    
    # [스탑로스 관련 변수]
    stoploss_active = False
    reentry_price = 0.0
    peak_assets = cash # 최고 자산 추적

    history, slot_details, trade_profits = [], [], []
    qqq_start_p = float(sim_df['qqq_close'].iloc[0])
    
    for date_idx, row in sim_df.iterrows():
        if ivy_reserve > 0: ivy_reserve *= (1 + boxx_rate)
        curr_c = row['close']
        
        # 1. 자산 평가 및 스탑로스 체크
        current_portfolio_val = cash + (shares * curr_c) + ivy_reserve
        if current_portfolio_val > peak_assets:
            peak_assets = current_portfolio_val
        
        # 현재 자산이 최고점 대비 -10% 하락 시 스탑로스 발동
        if not stoploss_active and shares > 0 and current_portfolio_val < (peak_assets * 0.90):
            stoploss_active = True
            reentry_price = curr_c
            # 전량 시장가 매도 (백테스트상 종가 매도)
            cash += (shares * curr_c)
            trade_profits.append((shares * curr_c) - (avg_price * shares))
            shares, used_slots, slot_cash, avg_price, slot_details = 0.0, 0, 0.0, 0.0, []

        # 2. 재진입 체크 (스탑로스 당시 가격을 돌파했는지)
        if stoploss_active and curr_c >= reentry_price:
            stoploss_active = False
            # 재진입 시점의 자산을 새로운 고점으로 리셋하여 다음 스탑로스 준비
            peak_assets = cash + ivy_reserve

        # 3. 전략 로직 (스탑로스 활성화 중에는 매매 금지)
        p1, p2, rsi_v = row['p1_c'], row['p2_c'], row['rsi']
        x = np.ceil(((p1 + p2) * 1.01 / 1.99) * 100) / 100
        
        if rsi_v > 65: mode, b_l, s_l = "Ivy", x - 0.01, np.ceil((x * 1.03) * 100) / 100
        elif rsi_v > 45: mode, b_l, s_l = "Willow", x - 0.01, x
        elif rsi_v > 30: mode, b_l, s_l = "Lily", np.floor((x * 0.975) * 100) / 100, x
        else: mode, b_l, s_l = "Tulip", np.floor((x * 0.975) * 100) / 100, x

        # 손절 방지 로직 (Break-even Only)
        effective_s_l = max(s_l, avg_price) if shares > 0 else s_l

        sold = False
        # 매도 로직 (스탑로스 상태와 무관하게 보유분이 있다면 조건 시 매도)
        if shares > 0 and curr_c >= effective_s_l:
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
        
        # 매수 로직 (스탑로스 활성 중에는 실행 안함)
        if not stoploss_active:
            if used_slots == 0 and float(pending_dep) != 0: # 실제 코드에선 이 부분은 history 누적 후에 처리되나 시뮬 내 일관성을 위해 유지
                pass # 외부에서 처리

            if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
                cash += ivy_reserve; ivy_reserve = 0.0
                
            if not sold and used_slots < (num_slots + 1) and curr_c <= b_l:
                if used_slots == 0: slot_cash = cash / num_slots
                current_order_cash = cash if used_slots >= num_slots else slot_cash
                buy_qty = current_order_cash // b_l 
                actual_cost = buy_qty * curr_c
                if buy_qty > 0 and cash >= actual_cost:
                    slot_label = "예비" if used_slots >= num_slots else used_slots + 1
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
            'QQQ': (initial_seed / qqq_start_p) * row['qqq_close'], 'Ivy': ivy_reserve,
            'Stoploss_Active': stoploss_active, 'Reentry_P': reentry_price
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
    p_dep = st.number_input("추가 입금/출금액 ($)", value=float(get_setting('pending_dep', 0.0)))

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
            cur = res_live.iloc[-1]
            valid_df = raw_df[raw_df.index.date < today_val]
            if valid_df.empty: valid_df = raw_df.iloc[:-1]
            
            if len(valid_df) >= 2:
                latest_closed_row = valid_df.iloc[-1]
                prev_closed_row = valid_df.iloc[-2]
                p1_val, p2_val = latest_closed_row['close'], prev_closed_row['close']
                rsi_val = latest_closed_row['rsi_live']
                data_date = valid_df.index[-1].strftime('%Y-%m-%d')
                
                zero_slots_df = res_live[res_live['Slots'] == 0]
                base_capital = zero_slots_df.iloc[-1]['Cash'] if not zero_slots_df.empty else init_seed
                fund_cycle = base_capital + p_dep

                st.subheader(f"📊 {target_ticker} 현재 운용 현황")
                st.caption(f"🕒 최종 업데이트 (KST): {now_kst} | 📅 가이드 계산 기준일: {data_date}")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("총 수익률", f"{(cur['Total']/init_seed-1)*100:+.2f}%")
                c2.metric("평균 단가", f"${cur['Avg']:.2f}")
                
                slot_display = f"{int(cur['Slots'])} / {num_slots}" if cur['Slots'] <= num_slots else f"{num_slots} + 예비"
                c3.metric("채워진 슬롯", slot_display)
                c4.metric("현재 창출 가치", f"${cur['Total']:,.2f}")
                c5.metric("사이클 기준금액", f"${fund_cycle:,.2f}")
                
                st.info(f"🏦 Ivy 비상금: **${cur['Ivy']:,.2f}** | 💸 PCR 인출액: **${cur['Withdrawn']:,.2f}**")
                
                if slots_live:
                    st.markdown("#### 📝 확정된 보유 슬롯 내역")
                    st.table(pd.DataFrame(slots_live))
                st.divider()
                
                # --- 오늘의 가이드 로직 (스탑로스 반영) ---
                x = np.ceil(((p1_val + p2_val) * 1.01 / 1.99) * 100) / 100
                mode, color = ("Ivy", "red") if rsi_val > 65 else ("Willow", "orange") if rsi_val > 45 else ("Lily", "blue") if rsi_val > 30 else ("Tulip", "purple")
                
                if cur['Stoploss_Active']:
                    st.error(f"### 🛑 시스템 경보: 자산 보호를 위한 매매 정지 중")
                    st.markdown(f"**이유:** 자산이 고점 대비 -10% 하락하여 스탑로스가 발동되었습니다.\n\n**재진입 조건:** {target_ticker} 주가가 **${cur['Reentry_P']:.2f}** 위로 올라가야 다시 매수가 시작됩니다. (현재가 관망)")
                else:
                    st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}])")
                    g1, g2 = st.columns(2)
                    with g1:
                        b_p = x - 0.01 if rsi_val > 45 else np.floor((x * 0.975)*100)/100
                        if cur['Slots'] < num_slots:
                            st.error(f"#### {int(cur['Slots'])+1}회차 정규 매수 (LOC)")
                            try:
                                current_fund = fund_cycle
                                if mode == "Tulip" and cur['Slots'] == 0: current_fund += cur['Ivy']
                                order_cash = current_fund / num_slots
                                st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(order_cash // b_p)} 주`")
                            except: st.write("⚠️ 계산 오류")
                        elif cur['Slots'] == num_slots:
                            st.warning(f"#### 🔥 예비 슬롯 추가 매수 (LOC)")
                            order_cash = cur['Cash']
                            st.write(f"**타점:** `${b_p:.2f}` 이하 | **정량:** `{int(order_cash // b_p)} 주`")
                        else: st.write("✅ 매수 완료")

                    with g2:
                        s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                        effective_s_p = max(s_p, cur['Avg']) if cur['Shares'] > 0 else s_p
                        st.info(f"#### 📤 전량 매도 (LOC)")
                        if cur['Shares'] > 0:
                            st.write(f"**타점:** `${effective_s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                        else: st.write("보유 없음")
                st.line_chart(res_live[['Total', 'QQQ']])

# --- 백테스트 및 도움말 ---
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
                f_val = res_b['Total'].iloc[-1]
                cagr = ((f_val / bt_seed) ** (365.25 / (res_b.index[-1] - res_b.index[0]).days) - 1) * 100
                mdd = (res_b['Total'] / res_b['Total'].cummax() - 1).min() * 100
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("최종 자산", f"${f_val:,.0f}"); m2.metric("CAGR", f"{cagr:.2f}%"); m3.metric("MDD", f"{mdd:.2f}%")
                st.line_chart(res_b[['Total', 'QQQ']])
                res_b['year'] = res_b.index.year
                y_stats = []
                for yr in sorted(res_b['year'].unique()):
                    y_df = res_b[res_b['year'] == yr]
                    y_stats.append({"연도": yr, "수익률": f"{(y_df['Total'].iloc[-1]/y_df['Total'].iloc[0]-1)*100:.1f}%", "MDD": f"{(y_df['Total']/y_df['Total'].cummax()-1).min()*100:.1f}%"})
                st.table(pd.DataFrame(y_stats))

with tab3:
    st.header("📖 사계절 전략 Pro 이용 가이드")
    info_category = st.radio("궁금한 항목을 선택하세요", ["⚡ 사용법 요약", "🛡️ 스탑로스 & 매매 정지", "🆕 예비 슬롯(+1)", "🛡️ 손절 방지 로직", "⚙️ 운용 설정", "📥 LOC 주문 가이드", "💰 수기 자금 관리"], horizontal=True)
    st.divider()
    if info_category == "⚡ 사용법 요약":
        st.subheader("🚀 핵심 사용법")
        st.markdown("1. 설정 저장 후 탭1 가이드 확인\n2. 가이드에 적힌 타점과 정량대로 증권사 앱에서 **LOC 주문** 예약\n3. 매일 밤 반복 (스탑로스 알람이 뜨면 회복 시까지 관망)")
    elif info_category == "🛡️ 스탑로스 & 매매 정지":
        st.subheader("🚨 자산 평가액 -10% 스탑로스")
        st.markdown("""1. **작동 조건**: 나의 전체 자산(현금+주식)이 최근 최고점 대비 **-10% 하락**하면 즉시 모든 주식을 매도합니다.
2. **매매 정지**: 스탑로스 매도 후에는 시장 관망 모드로 진입하며, 신규 매수를 중단합니다.
3. **재진입 조건**: 주가가 **스탑로스 당시의 종가 위로 다시 회복**될 때만 전략이 재가동됩니다.
4. **목적**: 예측 불가능한 대폭락장(Black Swan)에서 자산이 녹아내리는 것을 방지하기 위한 실험적 장치입니다.""")
    elif info_category == "🆕 예비 슬롯(+1)":
        st.subheader("🆕 예비 슬롯(+1) 안내")
        st.markdown("정규 슬롯(1~N) 매수 후에도 가격이 낮으면 남은 현금을 모두 털어넣어 단가를 최대로 낮추는 '보너스 한 발'입니다.")
    elif info_category == "🛡️ 손절 방지 로직":
        st.subheader("🛡️ 원금 사수: 무손실 매도")
        st.markdown("전략 매도 타점이 내 평단가보다 낮을 경우, 매도 타점을 내 평단가(본전)로 자동 상향합니다.")
    elif info_category == "⚙️ 운용 설정":
        st.subheader("운용 설정 설명")
        st.markdown("분할 수(4~5 추천), PCR(0.7~1 추천)을 통해 복리 효과와 안정성을 조절하세요.")
    elif info_category == "📥 LOC 주문 가이드":
        st.subheader("토스증권 LOC 주문")
        st.markdown("주문 유형을 **'LOC'**로 바꾼 뒤 타점과 수량을 넣고 예약하세요.")
    elif info_category == "💰 수기 자금 관리":
        st.subheader("입출금 관리")
        st.markdown("입금은 양수, 출금은 음수. 전량 매도 후 현금 상태일 때만 적용하세요.")
