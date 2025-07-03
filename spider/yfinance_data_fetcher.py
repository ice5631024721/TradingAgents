#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yfinance数据获取器
提供通用的金融数据获取接口，支持参数化配置
"""

import yfinance as yf
import pandas as pd
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YFinanceDataFetcher:
    """
    yfinance数据获取器类
    提供标准化的金融数据获取接口
    """
    
    def __init__(self, rate_limit_delay: float = 1.0, max_retries: int = 3):
        """
        初始化数据获取器
        
        Args:
            rate_limit_delay: 请求间隔时间（秒），用于控制API调用频率
            max_retries: 最大重试次数
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_request_time = 0
    
    def _apply_rate_limit(self):
        """应用频率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.info(f"应用频率限制，等待 {sleep_time:.2f} 秒")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_historical_data(
        self,
        ticker_name: str,
        start_date: str,
        end_date: str,
        time_interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取历史数据
        
        Args:
            ticker_name: 股票代码，如 'AAPL', 'TSLA'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            time_interval: 时间间隔，可选值：
                - '1m', '2m', '5m', '15m', '30m', '60m', '90m' (分钟级)
                - '1h' (小时级)
                - '1d', '5d' (日级)
                - '1wk' (周级)
                - '1mo', '3mo' (月级)
            **kwargs: 其他yfinance参数
        
        Returns:
            pandas.DataFrame: 包含OHLCV数据的DataFrame，如果失败返回None
        """
        # 验证输入参数
        if not self._validate_inputs(ticker_name, start_date, end_date, time_interval):
            return None
        
        logger.info(f"正在获取 {ticker_name} 从 {start_date} 到 {end_date} 的 {time_interval} 数据")
        
        for attempt in range(self.max_retries + 1):
            try:
                # 应用频率限制
                self._apply_rate_limit()

                yf.set_config("http://127.0.0.1:1087")

                # 创建Ticker对象
                ticker = yf.Ticker(ticker_name)

                # 获取历史数据
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=time_interval,
                    **kwargs
                )
                
                if data.empty:
                    logger.warning(f"未获取到 {ticker_name} 的数据")
                    return None
                
                # 数据清理和格式化
                data = self._clean_data(data)
                
                logger.info(f"成功获取 {len(data)} 条数据记录")
                return data
                
            except Exception as e:
                error_msg = str(e)
                if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < self.max_retries:
                        wait_time = (attempt + 1) * self.rate_limit_delay * 2
                        logger.warning(f"遇到频率限制，第 {attempt + 1} 次重试，等待 {wait_time:.2f} 秒")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"达到最大重试次数，获取数据失败: {error_msg}")
                        return None
                else:
                    logger.error(f"获取数据时发生错误: {error_msg}")
                    return None
        
        return None
    
    def _validate_inputs(
        self,
        ticker_name: str,
        start_date: str,
        end_date: str,
        time_interval: str
    ) -> bool:
        """
        验证输入参数
        
        Args:
            ticker_name: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            time_interval: 时间间隔
        
        Returns:
            bool: 验证是否通过
        """
        # 验证ticker_name
        if not ticker_name or not isinstance(ticker_name, str):
            logger.error("ticker_name必须是非空字符串")
            return False
        
        # 验证日期格式
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                logger.error("开始日期必须早于结束日期")
                return False
                
        except ValueError as e:
            logger.error(f"日期格式错误，请使用YYYY-MM-DD格式: {str(e)}")
            return False
        
        # 验证时间间隔
        valid_intervals = [
            '1m', '2m', '5m', '15m', '30m', '60m', '90m',
            '1h', '1d', '5d', '1wk', '1mo', '3mo'
        ]
        
        if time_interval not in valid_intervals:
            logger.error(f"无效的时间间隔: {time_interval}，有效值: {valid_intervals}")
            return False
        
        return True
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清理和格式化数据
        
        Args:
            data: 原始数据DataFrame
        
        Returns:
            pd.DataFrame: 清理后的数据
        """
        # 移除空值行
        data = data.dropna()
        
        # 确保列名标准化
        data.columns = [col.title() for col in data.columns]
        
        # 添加一些有用的计算列
        if 'High' in data.columns and 'Low' in data.columns:
            data['Range'] = data['High'] - data['Low']
        
        if 'Close' in data.columns and 'Open' in data.columns:
            data['Change'] = data['Close'] - data['Open']
            data['Change_Pct'] = (data['Change'] / data['Open']) * 100
        
        return data
    
    def get_ticker_info(self, ticker_name: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            ticker_name: 股票代码
        
        Returns:
            dict: 股票信息字典，如果失败返回None
        """
        for attempt in range(self.max_retries + 1):
            try:
                self._apply_rate_limit()
                
                ticker = yf.Ticker(ticker_name)
                info = ticker.info
                
                logger.info(f"成功获取 {ticker_name} 的基本信息")
                return info
                
            except Exception as e:
                error_msg = str(e)
                if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < self.max_retries:
                        wait_time = (attempt + 1) * self.rate_limit_delay * 2
                        logger.warning(f"遇到频率限制，第 {attempt + 1} 次重试，等待 {wait_time:.2f} 秒")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"达到最大重试次数，获取股票信息失败: {error_msg}")
                        return None
                else:
                    logger.error(f"获取股票信息时发生错误: {error_msg}")
                    return None
        
        return None


def fetch_historical_data(
    ticker_name: str,
    start_date: str,
    end_date: str,
    time_interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    便捷函数：获取历史数据
    
    Args:
        ticker_name: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        time_interval: 时间间隔
    
    Returns:
        pandas.DataFrame: 历史数据
    """
    fetcher = YFinanceDataFetcher()
    return fetcher.get_historical_data(ticker_name, start_date, end_date, time_interval)


if __name__ == "__main__":
    # 示例用法
    print("=== yfinance数据获取器测试 ===")
    
    # 测试参数
    ticker = "AAPL"
    start = "2025-01-01"
    end = "2025-07-01"
    interval = "1d"
    
    print(f"\n正在测试获取 {ticker} 从 {start} 到 {end} 的 {interval} 数据...")
    
    # 使用类方式
    fetcher = YFinanceDataFetcher(rate_limit_delay=2.0, max_retries=2)
    data = fetcher.get_historical_data(ticker, start, end, interval)
    
    if data is not None:
        print(f"\n成功获取数据，共 {len(data)} 条记录")
        print("\n数据预览:")
        print(data.head())
        print("\n数据列名:")
        print(data.columns.tolist())
        print("\n数据统计:")
        print(data.describe())
    else:
        print("数据获取失败")
    
    # 测试获取股票信息
    print(f"\n\n正在获取 {ticker} 的基本信息...")
    info = fetcher.get_ticker_info(ticker)
    
    if info:
        print(f"公司名称: {info.get('longName', 'N/A')}")
        print(f"行业: {info.get('industry', 'N/A')}")
        print(f"市值: {info.get('marketCap', 'N/A')}")
    
    print("\n=== 测试完成 ===")