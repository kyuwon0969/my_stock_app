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

        # [수정 로직] 손절 방지: 평단가가 전략 매도점보다 높으면 평단가를 매도점으로 설정
        effective_s_l = max(s_l, avg_price) if shares > 0 else s_l

        sold = False
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
        
        if used_slots == 0 and internal_pending != 0:
            cash += internal_pending
            internal_pending = 0.0

        if used_slots == 0 and mode == "Tulip" and ivy_reserve > 0:
            cash += ivy_reserve; ivy_reserve = 0.0
            
        if not sold and used_slots < (num_slots + 1) and curr_c <= b_l:
            if used_slots == 0: 
                slot_cash = cash / num_slots
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
            valid_df = raw_df[raw_df.index.date < today_val]
            if valid_df.empty: valid_df = raw_df.iloc[:-1]
            
            if len(valid_df) >= 2:
                latest_closed_row = valid_df.iloc[-1]
                prev_closed_row = valid_df.iloc[-2]
                p1_val, p2_val = latest_closed_row['close'], prev_closed_row['close']
                rsi_val = latest_closed_row['rsi_live']
                data_date = valid_df.index[-1].strftime('%Y-%m-%d')
                cur = res_live.iloc[-1]
                
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
                
                x = np.ceil(((p1_val + p2_val) * 1.01 / 1.99) * 100) / 100
                mode, color = ("Ivy", "red") if rsi_val > 65 else ("Willow", "orange") if rsi_val > 45 else ("Lily", "blue") if rsi_val > 30 else ("Tulip", "purple")
                tulip_msg = " (만약 Ivy 비상금으로 BOXX를 매수한 상태라면, BOXX를 현재 가격으로 전량 매도하세요.)" if mode == "Tulip" and cur['Slots'] == 0 else ""
                st.markdown(f"### 🎯 오늘의 실전 가이드 (현재 모드: :{color}[{mode}]{tulip_msg})")
                
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
                    else:
                        st.write("✅ 매수 완료")

                with g2:
                    s_p = np.ceil((x * 1.03)*100)/100 if rsi_val > 65 else x
                    # [수정 로직] 실시간 가이드에도 손절 방지 적용
                    effective_s_p = max(s_p, cur['Avg']) if cur['Shares'] > 0 else s_p
                    
                    st.info(f"#### 📤 전량 매도 (LOC)")
                    if cur['Shares'] > 0:
                        st.write(f"**타점:** `${effective_s_p:.2f}` 이상 | **수량:** `{int(cur['Shares'])} 주`")
                        if cur['Avg'] > s_p:
                            st.caption("🛡️ **손절 방지 작동 중**: 전략 매도 타점이 평단가보다 낮아, 본전 가격으로 주문을 올렸습니다.")
                    else: st.write("보유 없음")
                st.line_chart(res_live[['Total', 'QQQ']])

# --- 백테스트 탭 ---
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

# --- 도움말 탭 (복구 및 업데이트) ---
with tab3:
    st.header("📖 사계절 전략 Pro 이용 가이드")
    info_category = st.radio("궁금한 항목을 선택하세요", ["⚡ 사이트 사용법 3줄 요약", "🌿 Seasons 전략이란?", "🎯 실시간 현황 및 가이드 설명", "📊 백테스트 용어 설명", "⚙️ 운용 설정 설명", "🆕 예비 슬롯(+1) 로직 안내", "🛡️ 손절 방지 로직", "📥 LOC 주문 방법 (토스증권)", "💰 수기 자금 관리"], horizontal=True)
    st.divider()
    
    if info_category == "⚡ 사이트 사용법 3줄 요약":
        st.subheader("🚀 핵심 사용법 요약")
        st.markdown("""1. **설정하기**: 왼쪽 사이드바 '운용 설정'에서 분할 수(4 또는 5 추천), 실제 운용 시작일, 투자 원금(달러 기준), PCR(0.7에서 1 사이를 추천)을 설정하고 **'설정값 저장 및 강제 새로고침'** 버튼을 누른다.\n2. **확인하기**: '실시간 현황 & 가이드' 탭에서 **'오늘의 실전 가이드'**에 떠 있는 매수/매도 주문 가격과 수량을 확인한다.\n3. **주문하기**: 사용하는 증권 앱에서 그대로 달러 기준으로 **LOC 주문을 매일같이 건다**(휴장일 제외). (어렵다면 info 탭의 'LOC 주문 가이드' 카테고리 확인)""")
    elif info_category == "🌿 Seasons 전략이란?":
        st.subheader("1. 퀀트 투자(Quantitative Trading)란?")
        st.write("주식을 전혀 몰라도 괜찮습니다! 퀀트 투자는 사람의 감정이나 짐작 대신, **철저하게 '데이터'와 '규칙'에 따라 기계적으로 매매**하는 방식입니다.")
        st.subheader("2. Seasons 전략의 핵심")
        st.markdown("""* **자동 계산된 타점**: 과거 데이터를 분석해 최적의 매수/매도 가격을 매일 알려줍니다.\n* **예약 주문(LOC)**: 매일 밤 장이 마감될 때 설정한 가격이 오면 자동으로 거래가 체결되는 **LOC 주문**을 활용합니다.""")
        st.subheader("3. 4가지 운용 모드 설명")
        st.markdown("""시장 상황(QQQ RSI 지수)에 따라 전략은 4가지 모드로 자동 변신합니다.\n* **Ivy(아이비)**: 과열 상태. 보너스 수익금을 비상금으로 저축.\n* **Willow(윌로우)**: 안정 상태. 표준 매매 진행.\n* **Lily(릴리)**: 조정 상태. 낮은 가격 매수 대기.\n* **Tulip(튤립)**: 공포 상태. 비상금을 투입해 공격적 매수.""")
    elif info_category == "🎯 실시간 현황 및 가이드 설명":
        st.subheader("1. 주요 수치 및 위젯 설명")
        st.markdown("""* **총 수익률**: 원금 대비 현재 자산 성장률.\n* **평균 단가**: 보유 주식의 평균 매수 가격.\n* **채워진 슬롯**: 현재까지 진행된 매수 단계.\n* **사이클 기준금액**: 이번 회차 매매의 확정 원금.""")
    elif info_category == "📊 백테스트 용어 설명":
        st.subheader("과거 데이터 기반 백테스트란?")
        st.markdown("""* **CAGR**: 연평균 복리 수익률.\n* **MDD**: 전고점 대비 최대 하락폭 (낮을수록 안전).\n* **Calmar**: 하락 위험 대비 수익 효율 지표.""")
    elif info_category == "⚙️ 운용 설정 설명":
        st.subheader("전략 운용을 위한 핵심 설정")
        st.markdown("""* **매수 슬롯 분할 수**: 전체 투자금을 나누는 횟수 (4~5 추천).\n* **PCR (재투자 비중)**: 매도 수익 중 원금에 합칠 비율 (0.7~1 추천).""")
    elif info_category == "🆕 예비 슬롯(+1) 로직 안내":
        st.subheader("🆕 예비 슬롯(+1) 및 사이클 지표 안내")
        st.markdown("""1. **정규 슬롯 (1~N)**: 기준금액을 N등분하여 정량 매수합니다.\n2. **예비 슬롯 (N+1)**: 정규 매수 완료 후에도 하락하면, 남은 **자투리 현금 전량**을 투입합니다.\n3. **효과**: 현금 놀리는 구간을 최소화하고 하락장에서 단가 방어력을 극대화합니다.""")
    elif info_category == "🛡️ 손절 방지 로직":
        st.subheader("🛡️ 원금 사수: 무손실 매도 원칙")
        st.markdown("""1. **작동 원리**: 전략상 계산된 매도 타점이 나의 **평균 단가보다 낮을 경우**, 자동으로 매도 타점을 **내 평단가**로 올립니다.\n2. **장점**: 시장이 일시적으로 급락했을 때 기계적인 손절이 나가는 것을 방지하고, 최소 본전 반등 시까지 기다립니다.\n3. **주의**: 하락장이 길어질 경우 매도 타이밍이 늦어져 **보유 기간(MDD)이 길어질 수** 있습니다. 백테스트를 통해 이 리스크를 꼭 확인하세요.""")
    elif info_category == "📥 LOC 주문 방법 (토스증권)":
        st.subheader("토스증권 LOC 주문 방법")
        st.markdown("주문 종류를 **'LOC'**로 변경 후 가이드의 타점과 수량을 입력하여 예약합니다.")
    elif info_category == "💰 수기 자금 관리":
        st.subheader("입출금 및 자금 관리")
        st.markdown("추가 입금은 양수, 출금은 음수로 입력하세요. **전액 현금 상태**일 때 적용하는 것을 추천합니다.")
