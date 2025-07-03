#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率自适应动量策略 (Volatility-Adaptive Momentum, VAM)

策略核心逻辑：
1. 动量确认：连续3个周期收盘价高于20周期均线，且MACD柱状线持续放大
2. 波动率过滤：当前ATR > 历史90%分位数时触发策略
3. 量价背离修正：股价创新高但成交量低于前5周期均值时暂缓开仓

适用场景：高波动率市场（政策发布前后、财报季等）
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VAMStrategy:
    """
    波动率自适应动量策略实现类
    """
    
    def __init__(self, symbol='SPY', period='5m', lookback_days=30):
        """
        初始化策略参数
        
        Args:
            symbol: 交易标的
            period: K线周期（5m为5分钟）
            lookback_days: 回测天数
        """
        self.symbol = symbol
        self.period = period
        self.lookback_days = lookback_days
        
        # 策略参数
        self.ma_period = 20  # 均线周期
        self.momentum_periods = 3  # 动量确认周期数
        self.atr_period = 20  # ATR计算周期
        self.atr_percentile = 90  # ATR分位数阈值
        self.volume_periods = 5  # 成交量均值周期
        
        # MACD参数
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 交易参数
        self.initial_capital = 100000  # 初始资金
        self.position_size = 0.95  # 仓位大小
        self.stop_loss = 0.02  # 止损比例
        self.take_profit = 0.04  # 止盈比例
        
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None
        
    def fetch_data(self):
        """
        获取历史数据
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            print(f"正在获取 {self.symbol} 的 {self.period} 数据...")
            
            # 获取数据
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=f"{self.lookback_days}d", interval=self.period)
            
            if data.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据")
                
            # 重命名列名为中文（便于理解）
            data.columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
            
            print(f"成功获取 {len(data)} 条数据记录")
            print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")
            
            self.data = data
            return data
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            # 如果无法获取5分钟数据，尝试使用日线数据模拟
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        生成模拟数据用于测试
        """
        print("使用模拟数据进行测试...")
        
        # 生成基础价格序列
        np.random.seed(42)
        n_periods = self.lookback_days * 78  # 假设每天78个5分钟周期
        
        # 生成带趋势的随机游走
        returns = np.random.normal(0.0001, 0.02, n_periods)
        trend = np.linspace(0, 0.1, n_periods)  # 添加上升趋势
        returns += trend / n_periods
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        # 生成OHLC数据
        data = pd.DataFrame(index=pd.date_range(
            start=datetime.now() - timedelta(days=self.lookback_days),
            periods=n_periods,
            freq='5T'
        ))
        
        data['收盘价'] = prices
        data['开盘价'] = data['收盘价'].shift(1).fillna(data['收盘价'].iloc[0])
        
        # 生成高低价（基于收盘价的波动）
        volatility = np.random.uniform(0.005, 0.02, n_periods)
        data['最高价'] = data[['开盘价', '收盘价']].max(axis=1) * (1 + volatility)
        data['最低价'] = data[['开盘价', '收盘价']].min(axis=1) * (1 - volatility)
        
        # 生成成交量
        base_volume = 1000000
        volume_noise = np.random.lognormal(0, 0.5, n_periods)
        data['成交量'] = (base_volume * volume_noise).astype(int)
        
        self.data = data
        return data
    
    def calculate_indicators(self):
        """
        计算技术指标
        """
        if self.data is None:
            raise ValueError("请先获取数据")
            
        data = self.data.copy()
        
        # 1. 计算移动平均线
        data[f'MA{self.ma_period}'] = data['收盘价'].rolling(window=self.ma_period).mean()
        
        # 2. 计算MACD
        exp1 = data['收盘价'].ewm(span=self.macd_fast).mean()
        exp2 = data['收盘价'].ewm(span=self.macd_slow).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=self.macd_signal).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # 3. 计算ATR (真实波动幅度)
        data['TR'] = np.maximum(
            data['最高价'] - data['最低价'],
            np.maximum(
                abs(data['最高价'] - data['收盘价'].shift(1)),
                abs(data['最低价'] - data['收盘价'].shift(1))
            )
        )
        data['ATR'] = data['TR'].rolling(window=self.atr_period).mean()
        
        # 4. 计算ATR的历史分位数
        data['ATR_Percentile'] = data['ATR'].rolling(window=100).rank(pct=True) * 100
        
        # 5. 计算成交量均值
        data[f'Volume_MA{self.volume_periods}'] = data['成交量'].rolling(window=self.volume_periods).mean()
        
        # 6. 计算价格动量（连续上涨周期数）
        data['Price_Above_MA'] = (data['收盘价'] > data[f'MA{self.ma_period}']).astype(int)
        data['Momentum_Count'] = data['Price_Above_MA'].rolling(window=self.momentum_periods).sum()
        
        # 7. 计算MACD柱状线连续放大
        data['MACD_Hist_Increasing'] = (data['MACD_Hist'] > data['MACD_Hist'].shift(1)).astype(int)
        data['MACD_Momentum'] = data['MACD_Hist_Increasing'].rolling(window=self.momentum_periods).sum()
        
        # 8. 检测价格新高
        data['Price_High'] = data['收盘价'].rolling(window=20).max()
        data['Is_New_High'] = (data['收盘价'] >= data['Price_High']).astype(int)
        
        self.data = data
        return data
    
    def generate_signals(self):
        """
        生成交易信号
        """
        if self.data is None:
            raise ValueError("请先计算技术指标")
            
        data = self.data.copy()
        
        # 初始化信号
        data['Signal'] = 0
        data['Signal_Reason'] = ''
        
        for i in range(len(data)):
            if i < max(self.ma_period, self.atr_period, 100):  # 确保有足够的历史数据
                continue
                
            # 条件1: 动量确认
            momentum_confirmed = (
                data['Momentum_Count'].iloc[i] >= self.momentum_periods and
                data['MACD_Momentum'].iloc[i] >= self.momentum_periods
            )
            
            # 条件2: 波动率过滤
            volatility_filter = data['ATR_Percentile'].iloc[i] >= self.atr_percentile
            
            # 条件3: 量价背离修正
            volume_confirmation = True
            if data['Is_New_High'].iloc[i] == 1:
                volume_confirmation = data['成交量'].iloc[i] >= data[f'Volume_MA{self.volume_periods}'].iloc[i]
            
            # 生成买入信号
            if momentum_confirmed and volatility_filter and volume_confirmation:
                data.loc[data.index[i], 'Signal'] = 1
                data.loc[data.index[i], 'Signal_Reason'] = '动量+波动率+量价确认'
            
            # 生成卖出信号（简单的反向条件）
            elif (
                data['Momentum_Count'].iloc[i] == 0 or
                data['ATR_Percentile'].iloc[i] < 50
            ):
                data.loc[data.index[i], 'Signal'] = -1
                data.loc[data.index[i], 'Signal_Reason'] = '动量消失或波动率降低'
        
        self.signals = data
        return data
    
    def backtest(self):
        """
        执行回测
        """
        if self.signals is None:
            raise ValueError("请先生成交易信号")
            
        signals = self.signals.copy()
        
        # 初始化回测变量
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Price'] = signals['收盘价']
        portfolio['Signal'] = signals['Signal']
        portfolio['Position'] = 0
        portfolio['Holdings'] = 0
        portfolio['Cash'] = self.initial_capital
        portfolio['Total'] = self.initial_capital
        portfolio['Returns'] = 0
        portfolio['Strategy_Returns'] = 0
        
        # 交易执行
        position = 0
        cash = self.initial_capital
        entry_price = 0
        
        for i in range(1, len(portfolio)):
            current_price = portfolio['Price'].iloc[i]
            signal = portfolio['Signal'].iloc[i]
            
            # 检查止损止盈
            if position > 0 and entry_price > 0:
                # 止损检查
                if current_price <= entry_price * (1 - self.stop_loss):
                    # 止损卖出
                    cash += position * current_price * 0.999  # 扣除手续费
                    position = 0
                    entry_price = 0
                # 止盈检查
                elif current_price >= entry_price * (1 + self.take_profit):
                    # 止盈卖出
                    cash += position * current_price * 0.999
                    position = 0
                    entry_price = 0
            
            # 处理交易信号
            if signal == 1 and position == 0:  # 买入信号且无持仓
                shares_to_buy = (cash * self.position_size) / current_price
                if shares_to_buy > 0:
                    position = shares_to_buy
                    cash -= shares_to_buy * current_price * 1.001  # 扣除手续费
                    entry_price = current_price
                    
            elif signal == -1 and position > 0:  # 卖出信号且有持仓
                cash += position * current_price * 0.999
                position = 0
                entry_price = 0
            
            # 更新组合状态
            portfolio.loc[portfolio.index[i], 'Position'] = position
            portfolio.loc[portfolio.index[i], 'Holdings'] = position * current_price
            portfolio.loc[portfolio.index[i], 'Cash'] = cash
            portfolio.loc[portfolio.index[i], 'Total'] = cash + position * current_price
        
        # 计算收益率
        portfolio['Returns'] = portfolio['Price'].pct_change()
        portfolio['Strategy_Returns'] = portfolio['Total'].pct_change()
        
        self.portfolio = portfolio
        return portfolio
    
    def calculate_performance_metrics(self):
        """
        计算策略性能指标
        """
        if self.portfolio is None:
            raise ValueError("请先执行回测")
            
        portfolio = self.portfolio.dropna()
        
        # 基础指标
        total_return = (portfolio['Total'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率（假设252个交易日）
        trading_days = len(portfolio) / (252 * 78)  # 5分钟数据，每天78个周期
        annual_return = ((portfolio['Total'].iloc[-1] / self.initial_capital) ** (1/trading_days) - 1) * 100
        
        # 最大回撤
        rolling_max = portfolio['Total'].expanding().max()
        drawdown = (portfolio['Total'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # 夏普比率
        strategy_returns = portfolio['Strategy_Returns'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78)
        else:
            sharpe_ratio = 0
        
        # 胜率
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 最大连续亏损
        losses = strategy_returns[strategy_returns < 0]
        max_consecutive_losses = 0
        current_losses = 0
        for ret in strategy_returns:
            if ret < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        
        # 基准收益（买入持有）
        benchmark_return = (portfolio['Price'].iloc[-1] / portfolio['Price'].iloc[0] - 1) * 100
        
        metrics = {
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '夏普比率': round(sharpe_ratio, 2),
            '胜率(%)': round(win_rate, 2),
            '最大连续亏损次数': max_consecutive_losses,
            '基准收益率(%)': round(benchmark_return, 2),
            '超额收益(%)': round(total_return - benchmark_return, 2),
            '交易次数': total_trades,
            '最终资产': round(portfolio['Total'].iloc[-1], 2)
        }
        
        return metrics
    
    def plot_results(self):
        """
        绘制回测结果图表
        """
        if self.portfolio is None:
            raise ValueError("请先执行回测")
            
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. 价格走势和交易信号
        ax1 = axes[0]
        ax1.plot(self.portfolio.index, self.portfolio['Price'], label='价格', alpha=0.7)
        ax1.plot(self.signals.index, self.signals[f'MA{self.ma_period}'], label=f'MA{self.ma_period}', alpha=0.7)
        
        # 标记买卖点
        buy_signals = self.signals[self.signals['Signal'] == 1]
        sell_signals = self.signals[self.signals['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['收盘价'], color='green', marker='^', s=100, label='买入信号')
        ax1.scatter(sell_signals.index, sell_signals['收盘价'], color='red', marker='v', s=100, label='卖出信号')
        
        ax1.set_title('价格走势与交易信号')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 组合价值走势
        ax2 = axes[1]
        ax2.plot(self.portfolio.index, self.portfolio['Total'], label='策略组合价值', color='blue')
        
        # 基准线（买入持有）
        benchmark_value = self.initial_capital * (self.portfolio['Price'] / self.portfolio['Price'].iloc[0])
        ax2.plot(self.portfolio.index, benchmark_value, label='基准(买入持有)', color='orange', alpha=0.7)
        
        ax2.set_title('组合价值走势对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤分析
        ax3 = axes[2]
        rolling_max = self.portfolio['Total'].expanding().max()
        drawdown = (self.portfolio['Total'] - rolling_max) / rolling_max * 100
        ax3.fill_between(self.portfolio.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(self.portfolio.index, drawdown, color='red')
        ax3.set_title('策略回撤分析')
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 技术指标
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        # ATR和其分位数
        ax4.plot(self.signals.index, self.signals['ATR_Percentile'], label='ATR分位数', color='purple')
        ax4.axhline(y=self.atr_percentile, color='red', linestyle='--', label=f'{self.atr_percentile}%分位线')
        
        # MACD柱状线
        ax4_twin.bar(self.signals.index, self.signals['MACD_Hist'], alpha=0.3, label='MACD柱状线', color='blue')
        
        ax4.set_title('技术指标分析')
        ax4.set_ylabel('ATR分位数')
        ax4_twin.set_ylabel('MACD柱状线')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_strategy(self):
        """
        运行完整的策略流程
        """
        print("=" * 60)
        print("波动率自适应动量策略 (VAM) 回测系统")
        print("=" * 60)
        
        # 1. 获取数据
        print("\n1. 数据获取阶段")
        self.fetch_data()
        
        # 2. 计算指标
        print("\n2. 技术指标计算阶段")
        self.calculate_indicators()
        
        # 3. 生成信号
        print("\n3. 交易信号生成阶段")
        self.generate_signals()
        
        # 统计信号
        buy_signals = len(self.signals[self.signals['Signal'] == 1])
        sell_signals = len(self.signals[self.signals['Signal'] == -1])
        print(f"生成买入信号: {buy_signals} 个")
        print(f"生成卖出信号: {sell_signals} 个")
        
        # 4. 执行回测
        print("\n4. 回测执行阶段")
        self.backtest()
        
        # 5. 计算性能指标
        print("\n5. 性能分析阶段")
        metrics = self.calculate_performance_metrics()
        
        # 6. 输出结果
        print("\n" + "=" * 60)
        print("回测结果汇总")
        print("=" * 60)
        
        for key, value in metrics.items():
            print(f"{key:<20}: {value}")
        
        # 7. 绘制图表
        print("\n6. 生成可视化图表")
        self.plot_results()
        
        return metrics

if __name__ == "__main__":
    # 创建策略实例
    strategy = VAMStrategy(symbol='SPY', period='5m', lookback_days=30)
    
    # 运行策略
    results = strategy.run_strategy()
    
    print("\n策略运行完成！")
    print("图表已保存至: /Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_strategy_results.png")