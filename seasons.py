import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

class FourSeasons_Wilder_Backtester:
    def __init__(self, start_date, end_date, initial_seed=10000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_seed = initial_seed
        
        # 운영 변수 (5분할 균등 매수)
        self.cash = initial_seed
        self.shares = 0
        self.used_slots = 0
        self.num_slots = 5
        self.slot_cash = initial_seed / 5
        
        self.history = []

    def fetch_data(self):
        print(f"📡 데이터 수집 및 Wilder RSI 계산 중...")
        try:
            # Wilder 방식은 과거 데이터 예열이 중요하므로 6개월 전부터 수집
            fetch_start = pd.to_datetime(self.start_date) - pd.DateOffset(months=6)
            data = yf.download(["SOXL", "QQQ"], start=fetch_start, end=self.end_date)
            
            if isinstance(data.columns, pd.MultiIndex):
                soxl_all = data['Close']['SOXL'].dropna()
                qqq_all = data['Close']['QQQ'].dropna()
            else:
                soxl_all = data['SOXL'].dropna()
                qqq_all = data['QQQ'].dropna()

            # --- 트레이딩뷰 방식(Wilder's Smoothing) RSI 계산 ---
            delta = qqq_all.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # alpha = 1/14 설정이 Wilder 방식의 핵심
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            full_rsi = 100 - (100 / (1 + rs))
            
            # 시작일 이후 데이터로 필터링
            self.soxl_data = soxl_all[soxl_all.index >= self.start_date]
            self.rsi = full_rsi[full_rsi.index >= self.start_date]
            return self.soxl_data
        except Exception as e:
            print(f"❌ 데이터 오류: {e}")
            return None

    def run(self):
        df = self.soxl_data
        rsi = self.rsi
        
        for i in range(2, len(df)):
            p_prev1 = float(df.iloc[i-1])
            p_prev2 = float(df.iloc[i-2])
            curr_close = float(df.iloc[i])
            date = df.index[i]
            rsi_val = rsi.iloc[i-1] # 어제 확정 RSI 사용
            
            # 1. Willow_x 계산
            x_raw = (p_prev1 + p_prev2) * 1.01 / 1.99
            willow_x = np.ceil(x_raw * 100) / 100
            
            # 2. 사계절 모드별 타점 (친구의 최적화 로직 적용)
            if rsi_val > 65: # Ivy
                buy_limit = willow_x - 0.01
                sell_limit = np.ceil((willow_x * 1.03) * 100) / 100
            elif rsi_val > 45: # Willow
                buy_limit = willow_x - 0.01
                sell_limit = willow_x
            elif rsi_val > 30: # Lily
                buy_limit = np.floor((willow_x * 0.975) * 100) / 100
                sell_limit = willow_x
            else: # Tulip
                buy_limit = np.floor((willow_x * 0.975) * 100) / 100
                sell_limit = willow_x

            sold_today = False
            # 3. 매도 체크
            if self.shares > 0 and curr_close >= sell_limit:
                self.cash += self.shares * curr_close
                self.shares, self.used_slots = 0, 0
                self.slot_cash = self.cash / self.num_slots
                sold_today = True
            
            # 4. 매수 체크 (LOC 체결 시뮬레이션)
            if not sold_today and self.used_slots < self.num_slots and curr_close <= buy_limit:
                order_qty = self.slot_cash // buy_limit
                self.shares += order_qty
                self.cash -= (order_qty * curr_close)
                self.used_slots += 1
            
            self.history.append({'Date': date, 'Total': self.cash + (self.shares * curr_close)})

        self.results = pd.DataFrame(self.history).set_index('Date')
        self.report()

    def report(self):
        df = self.results
        final_val = df['Total'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        
        cagr = ((final_val / self.initial_seed) ** (1 / years) - 1) * 100
        mdd = (df['Total'] / df['Total'].cummax() - 1).min() * 100

        print(f"\n✨ [Wilder RSI 적용 결과] ✨")
        print(f"🏆 최종 자산: ${final_val:,.2f}")
        print(f"🚀 CAGR: {cagr:.2f}% | 📉 MDD: {mdd:.2f}%")
        
        # 그래프 출력
        plt.figure(figsize=(12, 6))
        plt.plot(df['Total'], color='darkgreen', label='Wilder Four Seasons')
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title("Performance with Wilder's RSI Smoothing")
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    tester = FourSeasons_Wilder_Backtester("2016-01-01", "2026-03-17")
    if tester.fetch_data() is not None:
        tester.run()
