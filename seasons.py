# --- 데이터 엔진 (시차 오류 수정) ---
@st.cache_data(ttl=60)
def get_processed_data(ticker, start_date):
    try:
        # 데이터 수집 (충분한 계산 기간 확보를 위해 종료일을 내일로 설정)
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
        
        # [수정] RSI 계산 로직: shift를 미리 하지 않고 계산 후 가이드에서 최신값 추출
        delta = df['qqq_close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        # 시뮬레이션용 RSI (전일 기준)
        df['rsi_sim'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).shift(1)
        # 가이드용 최신 RSI (오늘 매매 판단 기준)
        df['rsi_live'] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan)))))
        
        # p1, p2 데이터 생성
        df['p1_raw'] = df['close'].shift(0) # 오늘(가장 최근 종가)
        df['p2_raw'] = df['close'].shift(1) # 어제
        
        return df # dropna()를 여기서 하지 않고 시뮬레이션에서 처리
    except Exception as e:
        st.error(f"⚠️ 데이터 엔진 오류: {e}")
        return None
