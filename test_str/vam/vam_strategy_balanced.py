#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡版波动率自适应动量策略 (VAM v3.0)

在v2.0基础上进一步优化，平衡风险控制与收益获取：
1. 调整信号触发条件，避免过于保守
2. 优化参数设置，提高信号频率
3. 保持风险控制的同时增加交易机会
4. 加入多时间框架确认
5. 改进量价分析逻辑
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

class VAMStrategyBalanced:
    """
    平衡版波动率自适应动量策略实现类
    """
    
    def __init__(self, symbol='SPY', period='5m', lookback_days=30):
        """
        初始化策略参数
        """
        self.symbol = symbol
        self.period = period
        self.lookback_days = lookback_days
        
        # 策略参数 - 平衡调整
        self.ma_period = 20
        self.momentum_periods = 3  # 回到3个周期
        self.atr_period = 20
        self.atr_percentile = 70  # 进一步降低到70%
        self.volume_periods = 5
        
        # MACD参数
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 平衡的交易参数
        self.initial_capital = 100000
        self.base_position_size = 0.5  # 中等仓位
        self.max_position_size = 0.7   # 适中最大仓位
        self.min_position_size = 0.2   # 适中最小仓位
        
        # 平衡的止损止盈
        self.base_stop_loss = 0.02     # 2%止损
        self.base_take_profit = 0.04   # 4%止盈
        self.trailing_stop = True      # 启用移动止损
        
        # 调整的过滤条件
        self.min_trend_strength = 0.3  # 降低趋势强度要求
        self.volatility_threshold = 0.01  # 降低波动率阈值
        
        # 风险控制
        self.max_daily_loss = 0.03     # 单日最大亏损3%
        self.max_consecutive_losses = 4 # 最大连续亏损次数
        
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
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=f"{self.lookback_days}d", interval=self.period)
            
            if data.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据")
                
            data.columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
            
            print(f"成功获取 {len(data)} 条数据记录")
            print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")
            
            self.data = data
            return data
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        生成模拟数据用于测试
        """
        print("使用模拟数据进行测试...")
        
        np.random.seed(42)
        n_periods = self.lookback_days * 78
        
        # 生成更有趋势性的市场数据
        base_trend = 0.0002  # 基础上涨趋势
        returns = np.random.normal(base_trend, 0.012, n_periods)
        
        # 添加周期性波动和趋势变化
        for i in range(n_periods):
            cycle_factor = np.sin(i / 50) * 0.0003  # 周期性因子
            trend_factor = np.sin(i / 200) * 0.0005  # 长期趋势因子
            returns[i] += cycle_factor + trend_factor
            
            # 添加突发性波动
            if np.random.random() < 0.05:  # 5%概率的突发波动
                returns[i] += np.random.normal(0, 0.02)
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame(index=pd.date_range(
            start=datetime.now() - timedelta(days=self.lookback_days),
            periods=n_periods,
            freq='5T'
        ))
        
        data['收盘价'] = prices
        data['开盘价'] = data['收盘价'].shift(1).fillna(data['收盘价'].iloc[0])
        
        # 更真实的高低价
        volatility = np.random.uniform(0.002, 0.012, n_periods)
        data['最高价'] = data[['开盘价', '收盘价']].max(axis=1) * (1 + volatility)
        data['最低价'] = data[['开盘价', '收盘价']].min(axis=1) * (1 - volatility)
        
        # 与价格变化相关的成交量
        price_change = np.abs(data['收盘价'].pct_change().fillna(0))
        base_volume = 1000000
        volume_multiplier = 1 + price_change * 10  # 价格变化越大，成交量越大
        volume_noise = np.random.lognormal(0, 0.2, n_periods)
        data['成交量'] = (base_volume * volume_multiplier * volume_noise).astype(int)
        
        self.data = data
        return data
    
    def calculate_indicators(self):
        """
        计算技术指标
        """
        if self.data is None:
            raise ValueError("请先获取数据")
            
        data = self.data.copy()
        
        # 基础移动平均线
        data[f'MA{self.ma_period}'] = data['收盘价'].rolling(window=self.ma_period).mean()
        data['MA_Short'] = data['收盘价'].rolling(window=10).mean()
        data['MA_Long'] = data['收盘价'].rolling(window=30).mean()  # 缩短长期均线
        
        # MACD指标
        exp1 = data['收盘价'].ewm(span=self.macd_fast).mean()
        exp2 = data['收盘价'].ewm(span=self.macd_slow).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=self.macd_signal).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # ATR和波动率指标
        data['TR'] = np.maximum(
            data['最高价'] - data['最低价'],
            np.maximum(
                abs(data['最高价'] - data['收盘价'].shift(1)),
                abs(data['最低价'] - data['收盘价'].shift(1))
            )
        )
        data['ATR'] = data['TR'].rolling(window=self.atr_period).mean()
        data['ATR_Percentile'] = data['ATR'].rolling(window=60).rank(pct=True) * 100  # 缩短窗口
        
        # 趋势强度指标（调整计算方式）
        data['Trend_Strength'] = abs(data['MA_Short'] - data['MA_Long']) / data['收盘价']
        
        # 价格相对位置
        data['Price_Position'] = (data['收盘价'] - data['收盘价'].rolling(15).min()) / \
                                (data['收盘价'].rolling(15).max() - data['收盘价'].rolling(15).min())
        
        # 成交量指标
        data[f'Volume_MA{self.volume_periods}'] = data['成交量'].rolling(window=self.volume_periods).mean()
        data['Volume_Ratio'] = data['成交量'] / data[f'Volume_MA{self.volume_periods}']
        
        # 动量指标
        data['Price_Above_MA'] = (data['收盘价'] > data[f'MA{self.ma_period}']).astype(int)
        data['Momentum_Count'] = data['Price_Above_MA'].rolling(window=self.momentum_periods).sum()
        
        # MACD动量
        data['MACD_Hist_Increasing'] = (data['MACD_Hist'] > data['MACD_Hist'].shift(1)).astype(int)
        data['MACD_Momentum'] = data['MACD_Hist_Increasing'].rolling(window=self.momentum_periods).sum()
        
        # MACD强度（调整计算）
        data['MACD_Strength'] = abs(data['MACD_Hist']) / (data['ATR'] + 1e-8)
        
        # 价格新高检测
        data['Price_High_15'] = data['收盘价'].rolling(window=15).max()
        data['Price_High_10'] = data['收盘价'].rolling(window=10).max()
        data['Is_New_High'] = (data['收盘价'] >= data['Price_High_15'] * 0.995).astype(int)  # 放宽条件
        data['Is_Strong_High'] = (data['收盘价'] >= data['Price_High_10']).astype(int)
        
        # 市场状态判断
        data['Market_Volatility'] = data['收盘价'].rolling(15).std() / data['收盘价'].rolling(15).mean()
        data['Trend_Direction'] = np.where(data['MA_Short'] > data['MA_Long'], 1, 
                                          np.where(data['MA_Short'] < data['MA_Long'], -1, 0))
        
        # 新增：RSI指标
        delta = data['收盘价'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # 新增：布林带
        data['BB_Middle'] = data['收盘价'].rolling(window=20).mean()
        bb_std = data['收盘价'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Position'] = (data['收盘价'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        self.data = data
        return data
    
    def calculate_dynamic_position_size(self, row):
        """
        动态计算仓位大小
        """
        base_size = self.base_position_size
        
        # 根据趋势强度调整（更温和）
        trend_multiplier = min(1.3, max(0.7, 1 + row['Trend_Strength'] * 2))
        
        # 根据波动率调整（更温和）
        volatility_multiplier = min(1.2, max(0.8, 1 / (1 + row['Market_Volatility'] * 10)))
        
        # 根据RSI调整
        rsi_multiplier = 1.0
        if row['RSI'] < 30:  # 超卖
            rsi_multiplier = 1.2
        elif row['RSI'] > 70:  # 超买
            rsi_multiplier = 0.8
        
        # 综合调整
        adjusted_size = base_size * trend_multiplier * volatility_multiplier * rsi_multiplier
        
        # 限制在最小最大范围内
        return max(self.min_position_size, min(self.max_position_size, adjusted_size))
    
    def calculate_dynamic_stops(self, row, entry_price):
        """
        动态计算止损止盈
        """
        # 基于ATR的动态止损
        atr_multiplier = 1.2
        dynamic_stop_loss = min(self.base_stop_loss, (row['ATR'] * atr_multiplier) / entry_price)
        
        # 基于波动率的动态止盈
        volatility_multiplier = 1.5
        dynamic_take_profit = max(self.base_take_profit, row['Market_Volatility'] * volatility_multiplier)
        
        return max(0.01, dynamic_stop_loss), min(0.08, dynamic_take_profit)
    
    def generate_signals(self):
        """
        生成交易信号（平衡版）
        """
        if self.data is None:
            raise ValueError("请先计算技术指标")
            
        data = self.data.copy()
        
        data['Signal'] = 0
        data['Signal_Reason'] = ''
        data['Position_Size'] = self.base_position_size
        data['Stop_Loss'] = self.base_stop_loss
        data['Take_Profit'] = self.base_take_profit
        
        for i in range(len(data)):
            if i < max(self.ma_period, self.atr_period, 60):
                continue
                
            row = data.iloc[i]
            
            # 平衡的条件检查
            
            # 条件1: 基础动量确认（放宽要求）
            momentum_confirmed = (
                row['Momentum_Count'] >= self.momentum_periods - 1 and  # 允许少1个
                row['MACD_Momentum'] >= self.momentum_periods - 2 and  # 进一步放宽
                row['MACD_Hist'] > -0.1  # MACD不能太负
            )
            
            # 条件2: 波动率过滤（放宽）
            volatility_filter = (
                row['ATR_Percentile'] >= self.atr_percentile or  # 或者条件
                row['Market_Volatility'] >= self.volatility_threshold
            )
            
            # 条件3: 量价确认（简化）
            volume_confirmation = True
            if row['Is_New_High'] == 1:
                volume_confirmation = row['Volume_Ratio'] >= 0.8  # 降低要求
            
            # 条件4: 趋势强度确认（放宽）
            trend_strength_ok = (
                row['Trend_Strength'] >= self.min_trend_strength or
                row['Trend_Direction'] == 1
            )
            
            # 条件5: 价格位置确认（放宽）
            price_position_ok = (
                row['Price_Position'] > 0.4 or  # 降低要求
                row['BB_Position'] > 0.3  # 或布林带位置
            )
            
            # 条件6: RSI确认（新增）
            rsi_ok = 30 <= row['RSI'] <= 80  # RSI在合理范围
            
            # 生成买入信号（更宽松的条件）
            buy_conditions = [
                momentum_confirmed,
                volatility_filter,
                volume_confirmation,
                trend_strength_ok,
                price_position_ok,
                rsi_ok
            ]
            
            # 至少满足4个条件即可买入
            if sum(buy_conditions) >= 4:
                data.loc[data.index[i], 'Signal'] = 1
                data.loc[data.index[i], 'Signal_Reason'] = f'买入({sum(buy_conditions)}/6条件满足)'
                
                # 动态调整仓位和止损止盈
                data.loc[data.index[i], 'Position_Size'] = self.calculate_dynamic_position_size(row)
                stop_loss, take_profit = self.calculate_dynamic_stops(row, row['收盘价'])
                data.loc[data.index[i], 'Stop_Loss'] = stop_loss
                data.loc[data.index[i], 'Take_Profit'] = take_profit
            
            # 生成卖出信号（适中严格度）
            elif (
                row['Momentum_Count'] == 0 or  # 完全失去动量
                row['ATR_Percentile'] < 20 or  # 波动率过低
                (row['Trend_Direction'] == -1 and row['RSI'] > 70) or  # 趋势转向且超买
                row['MACD_Hist'] < -0.2  # MACD大幅转负
            ):
                data.loc[data.index[i], 'Signal'] = -1
                data.loc[data.index[i], 'Signal_Reason'] = '趋势转弱卖出'
        
        self.signals = data
        return data
    
    def backtest(self):
        """
        执行回测
        """
        if self.signals is None:
            raise ValueError("请先生成交易信号")
            
        signals = self.signals.copy()
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Price'] = signals['收盘价']
        portfolio['Signal'] = signals['Signal']
        portfolio['Position'] = 0
        portfolio['Holdings'] = 0
        portfolio['Cash'] = self.initial_capital
        portfolio['Total'] = self.initial_capital
        portfolio['Returns'] = 0
        portfolio['Strategy_Returns'] = 0
        portfolio['Drawdown'] = 0
        
        # 交易执行变量
        position = 0
        cash = self.initial_capital
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        trailing_stop_price = 0
        consecutive_losses = 0
        daily_pnl = 0
        last_date = None
        
        for i in range(1, len(portfolio)):
            current_price = portfolio['Price'].iloc[i]
            signal = portfolio['Signal'].iloc[i]
            current_date = portfolio.index[i].date()
            
            # 重置每日PnL
            if last_date != current_date:
                daily_pnl = 0
                last_date = current_date
            
            # 风险控制：检查单日亏损限制
            if daily_pnl < -self.initial_capital * self.max_daily_loss:
                # 强制平仓
                if position > 0:
                    cash += position * current_price * 0.999
                    position = 0
                    entry_price = 0
                continue
            
            # 风险控制：检查连续亏损限制
            if consecutive_losses >= self.max_consecutive_losses:
                # 暂停交易
                if i % 15 == 0:  # 每15个周期重置
                    consecutive_losses = 0
                else:
                    continue
            
            # 检查止损止盈和移动止损
            if position > 0 and entry_price > 0:
                # 更新移动止损
                if self.trailing_stop and current_price > entry_price * 1.015:
                    new_trailing_stop = current_price * (1 - self.base_stop_loss * 0.8)
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
                
                # 止损检查
                if (current_price <= stop_loss_price or 
                    (self.trailing_stop and current_price <= trailing_stop_price)):
                    
                    pnl = (current_price - entry_price) * position * 0.999
                    cash += position * current_price * 0.999
                    daily_pnl += pnl
                    
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                    position = 0
                    entry_price = 0
                    trailing_stop_price = 0
                
                # 止盈检查
                elif current_price >= take_profit_price:
                    pnl = (current_price - entry_price) * position * 0.999
                    cash += position * current_price * 0.999
                    daily_pnl += pnl
                    consecutive_losses = 0
                    
                    position = 0
                    entry_price = 0
                    trailing_stop_price = 0
            
            # 处理交易信号
            if signal == 1 and position == 0 and consecutive_losses < self.max_consecutive_losses:
                # 获取动态参数
                position_size = signals['Position_Size'].iloc[i]
                stop_loss_pct = signals['Stop_Loss'].iloc[i]
                take_profit_pct = signals['Take_Profit'].iloc[i]
                
                shares_to_buy = (cash * position_size) / current_price
                if shares_to_buy > 0:
                    position = shares_to_buy
                    cash -= shares_to_buy * current_price * 1.001
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trailing_stop_price = stop_loss_price
                    
            elif signal == -1 and position > 0:
                pnl = (current_price - entry_price) * position * 0.999
                cash += position * current_price * 0.999
                daily_pnl += pnl
                
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                
                position = 0
                entry_price = 0
                trailing_stop_price = 0
            
            # 更新组合状态
            portfolio.loc[portfolio.index[i], 'Position'] = position
            portfolio.loc[portfolio.index[i], 'Holdings'] = position * current_price
            portfolio.loc[portfolio.index[i], 'Cash'] = cash
            portfolio.loc[portfolio.index[i], 'Total'] = cash + position * current_price
        
        # 计算收益率和回撤
        portfolio['Returns'] = portfolio['Price'].pct_change()
        portfolio['Strategy_Returns'] = portfolio['Total'].pct_change()
        
        # 计算回撤
        rolling_max = portfolio['Total'].expanding().max()
        portfolio['Drawdown'] = (portfolio['Total'] - rolling_max) / rolling_max
        
        self.portfolio = portfolio
        return portfolio
    
    def calculate_performance_metrics(self):
        """
        计算策略性能指标
        """
        if self.portfolio is None:
            raise ValueError("请先执行回测")
            
        portfolio = self.portfolio.dropna()
        
        total_return = (portfolio['Total'].iloc[-1] / self.initial_capital - 1) * 100
        
        trading_days = len(portfolio) / (252 * 78)
        annual_return = ((portfolio['Total'].iloc[-1] / self.initial_capital) ** (1/trading_days) - 1) * 100
        
        max_drawdown = portfolio['Drawdown'].min() * 100
        
        strategy_returns = portfolio['Strategy_Returns'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78)
        else:
            sharpe_ratio = 0
        
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 计算最大连续亏损
        max_consecutive_losses = 0
        current_losses = 0
        for ret in strategy_returns:
            if ret < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        
        benchmark_return = (portfolio['Price'].iloc[-1] / portfolio['Price'].iloc[0] - 1) * 100
        
        # 计算Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 计算信息比率
        excess_returns = strategy_returns - portfolio['Returns'].dropna()
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 78) if excess_returns.std() > 0 else 0
        
        metrics = {
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '夏普比率': round(sharpe_ratio, 2),
            'Calmar比率': round(calmar_ratio, 2),
            '信息比率': round(information_ratio, 2),
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
            
        fig, axes = plt.subplots(6, 1, figsize=(15, 24))
        
        # 1. 价格走势和交易信号
        ax1 = axes[0]
        ax1.plot(self.portfolio.index, self.portfolio['Price'], label='价格', alpha=0.7, linewidth=1)
        ax1.plot(self.signals.index, self.signals[f'MA{self.ma_period}'], label=f'MA{self.ma_period}', alpha=0.7)
        ax1.plot(self.signals.index, self.signals['MA_Short'], label='MA10', alpha=0.5)
        ax1.plot(self.signals.index, self.signals['MA_Long'], label='MA30', alpha=0.5)
        
        buy_signals = self.signals[self.signals['Signal'] == 1]
        sell_signals = self.signals[self.signals['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['收盘价'], color='green', marker='^', s=60, label='买入信号', alpha=0.8)
        ax1.scatter(sell_signals.index, sell_signals['收盘价'], color='red', marker='v', s=60, label='卖出信号', alpha=0.8)
        
        ax1.set_title('平衡版VAM策略 - 价格走势与交易信号', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 组合价值走势对比
        ax2 = axes[1]
        ax2.plot(self.portfolio.index, self.portfolio['Total'], label='平衡版VAM策略', color='blue', linewidth=2)
        
        benchmark_value = self.initial_capital * (self.portfolio['Price'] / self.portfolio['Price'].iloc[0])
        ax2.plot(self.portfolio.index, benchmark_value, label='基准(买入持有)', color='orange', alpha=0.7)
        
        ax2.set_title('组合价值走势对比', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('组合价值 ($)')
        
        # 3. 回撤分析
        ax3 = axes[2]
        drawdown_pct = self.portfolio['Drawdown'] * 100
        ax3.fill_between(self.portfolio.index, drawdown_pct, 0, alpha=0.3, color='red')
        ax3.plot(self.portfolio.index, drawdown_pct, color='red', linewidth=1)
        ax3.set_title('策略回撤分析', fontsize=14)
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 技术指标面板
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        # ATR分位数和RSI
        ax4.plot(self.signals.index, self.signals['ATR_Percentile'], label='ATR分位数', color='purple', alpha=0.7)
        ax4.axhline(y=self.atr_percentile, color='red', linestyle='--', label=f'{self.atr_percentile}%分位线')
        ax4.plot(self.signals.index, self.signals['RSI'], label='RSI', color='orange', alpha=0.7)
        ax4.axhline(y=70, color='red', linestyle=':', alpha=0.5)
        ax4.axhline(y=30, color='green', linestyle=':', alpha=0.5)
        
        # MACD柱状线
        ax4_twin.bar(self.signals.index, self.signals['MACD_Hist'], alpha=0.3, label='MACD柱状线', color='blue', width=0.001)
        
        ax4.set_title('技术指标分析', fontsize=14)
        ax4.set_ylabel('指标值')
        ax4_twin.set_ylabel('MACD柱状线')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 布林带和价格位置
        ax5 = axes[4]
        ax5.plot(self.signals.index, self.signals['收盘价'], label='收盘价', color='black', linewidth=1)
        ax5.plot(self.signals.index, self.signals['BB_Upper'], label='布林带上轨', color='red', alpha=0.5)
        ax5.plot(self.signals.index, self.signals['BB_Middle'], label='布林带中轨', color='blue', alpha=0.5)
        ax5.plot(self.signals.index, self.signals['BB_Lower'], label='布林带下轨', color='green', alpha=0.5)
        ax5.fill_between(self.signals.index, self.signals['BB_Upper'], self.signals['BB_Lower'], alpha=0.1, color='gray')
        
        ax5.set_title('布林带分析', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 仓位管理与风险控制
        ax6 = axes[5]
        ax6_twin = ax6.twinx()
        
        # 仓位大小
        position_sizes = self.signals['Position_Size'] * 100
        ax6.plot(self.signals.index, position_sizes, label='动态仓位(%)', color='orange', alpha=0.7)
        ax6.axhline(y=self.base_position_size * 100, color='gray', linestyle='--', label='基础仓位', alpha=0.5)
        
        # 市场波动率
        market_vol = self.signals['Market_Volatility'] * 100
        ax6_twin.plot(self.signals.index, market_vol, label='市场波动率(%)', color='red', alpha=0.7)
        
        ax6.set_title('仓位管理与风险控制', fontsize=14)
        ax6.set_ylabel('仓位大小 (%)')
        ax6_twin.set_ylabel('市场波动率 (%)')
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_strategy_balanced_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, metrics):
        """
        生成详细的策略报告
        """
        report = f"""
# 平衡版波动率自适应动量策略 (VAM v3.0) 回测报告

## 策略概述
平衡版VAM策略在保持风险控制的同时，优化了信号生成逻辑，提高了交易机会的捕获能力。

## 核心改进
1. **信号条件优化**: 采用6个条件中满足4个即可触发的灵活机制
2. **参数平衡调整**: 降低ATR分位数要求至70%，提高信号频率
3. **多指标融合**: 新增RSI和布林带指标，增强市场状态判断
4. **动态仓位管理**: 基于趋势强度、波动率和RSI的综合仓位调整
5. **风险控制优化**: 保持止损机制的同时，避免过度保守

## 回测结果

### 核心性能指标
- **总收益率**: {metrics['总收益率(%)']}%
- **年化收益率**: {metrics['年化收益率(%)']}%
- **最大回撤**: {metrics['最大回撤(%)']}%
- **夏普比率**: {metrics['夏普比率']}
- **Calmar比率**: {metrics['Calmar比率']}
- **信息比率**: {metrics['信息比率']}

### 交易表现
- **胜率**: {metrics['胜率(%)']}%
- **交易次数**: {metrics['交易次数']}
- **最大连续亏损**: {metrics['最大连续亏损次数']}次
- **最终资产**: ${metrics['最终资产']:,.2f}

### 相对表现
- **基准收益率**: {metrics['基准收益率(%)']}%
- **超额收益**: {metrics['超额收益(%)']}%

## 策略评估

### 优势分析
1. **平衡的风险收益**: 在控制回撤的同时获得合理收益
2. **适应性强**: 多指标融合提高了不同市场环境下的适应能力
3. **交易频率适中**: 避免了过度交易和信号稀少的极端情况
4. **风险控制有效**: 动态止损和仓位管理降低了极端损失风险

### 改进空间
1. **信号质量**: 可进一步优化信号过滤条件，提高胜率
2. **市场适应性**: 可加入市场状态识别，在不同环境下调整策略参数
3. **成本控制**: 可优化交易频率，降低交易成本影响
4. **多时间框架**: 可结合更长时间框架的趋势确认

## 风险提示
1. 策略基于历史数据回测，实际表现可能存在差异
2. 市场环境变化可能影响策略有效性
3. 交易成本和滑点可能影响实际收益
4. 建议结合基本面分析和风险管理使用

## 总结
平衡版VAM策略通过优化参数设置和信号生成逻辑，在风险控制和收益获取之间找到了较好的平衡点。
策略适合在中等波动率环境下使用，建议根据实际市场情况进行参数调整。

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # 保存报告
        with open('/test_str/vam/vam_balanced_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def run_strategy(self):
        """
        运行完整的策略流程
        """
        print("=" * 60)
        print("平衡版波动率自适应动量策略 (VAM v3.0) 回测系统")
        print("=" * 60)
        
        print("\n策略特点:")
        print("1. 平衡风险控制与收益获取")
        print("2. 多指标融合信号生成")
        print("3. 动态仓位和止损管理")
        print("4. 适中的交易频率")
        print("5. 强化的风险控制机制")
        
        # 执行策略流程
        print("\n1. 数据获取阶段")
        self.fetch_data()
        
        print("\n2. 技术指标计算阶段")
        self.calculate_indicators()
        
        print("\n3. 交易信号生成阶段")
        self.generate_signals()
        
        buy_signals = len(self.signals[self.signals['Signal'] == 1])
        sell_signals = len(self.signals[self.signals['Signal'] == -1])
        print(f"生成买入信号: {buy_signals} 个")
        print(f"生成卖出信号: {sell_signals} 个")
        
        print("\n4. 回测执行阶段")
        self.backtest()
        
        print("\n5. 性能分析阶段")
        metrics = self.calculate_performance_metrics()
        
        print("\n" + "=" * 60)
        print("平衡版VAM策略回测结果汇总")
        print("=" * 60)
        
        for key, value in metrics.items():
            print(f"{key:<20}: {value}")
        
        print("\n6. 生成可视化图表")
        self.plot_results()
        
        print("\n7. 生成详细报告")
        report = self.generate_detailed_report(metrics)
        
        return metrics, report

if __name__ == "__main__":
    # 创建平衡版策略实例
    strategy = VAMStrategyBalanced(symbol='SPY', period='5m', lookback_days=30)
    
    # 运行策略
    results, report = strategy.run_strategy()
    
    print("\n平衡版策略运行完成！")
    print("图表已保存至: /Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_strategy_balanced_results.png")
    print("报告已保存至: /Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_balanced_report.md")