# YFinance 数据获取器使用指南

## 概述

本模块提供了一个标准化的 yfinance 数据获取接口，支持获取各种金融产品的历史数据。该接口具有以下特点：

- 🚀 **通用性强**: 支持股票、ETF、指数等多种金融产品
- 🛡️ **错误处理**: 完善的错误处理和重试机制
- ⏱️ **频率控制**: 内置API频率限制处理
- 📊 **数据清理**: 自动数据清理和格式化
- 🔧 **易于使用**: 简洁的API接口设计

## 安装依赖

确保已安装所需的依赖包：

```bash
pip install yfinance pandas
```

## 基本使用

### 1. 导入模块

```python
from spider.yfinance_data_fetcher import YFinanceDataFetcher, fetch_historical_data
```

### 2. 使用类接口（推荐）

```python
# 创建数据获取器实例
fetcher = YFinanceDataFetcher(
    rate_limit_delay=2.0,  # 请求间隔（秒）
    max_retries=3          # 最大重试次数
)

# 获取历史数据
data = fetcher.get_historical_data(
    ticker_name="AAPL",
    start_date="2024-01-01",
    end_date="2024-01-31",
    time_interval="1d"
)

if data is not None:
    print(f"成功获取 {len(data)} 条数据")
    print(data.head())
else:
    print("数据获取失败")
```

### 3. 使用便捷函数

```python
# 快速获取数据
data = fetch_historical_data(
    ticker_name="TSLA",
    start_date="2024-01-01",
    end_date="2024-01-31",
    time_interval="1d"
)
```

## 参数说明

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `ticker_name` | str | 股票代码 | "AAPL", "TSLA", "SPY" |
| `start_date` | str | 开始日期 | "2024-01-01" |
| `end_date` | str | 结束日期 | "2024-01-31" |
| `time_interval` | str | 时间间隔 | "1d", "1h", "5m" |

### 时间间隔选项

| 间隔 | 说明 | 数据保留期 |
|------|------|------------|
| `1m`, `2m`, `5m` | 分钟级 | 最近30天 |
| `15m`, `30m`, `60m`, `90m` | 分钟级 | 最近60天 |
| `1h` | 小时级 | 最近730天 |
| `1d` | 日级 | 无限制 |
| `5d` | 5日级 | 无限制 |
| `1wk` | 周级 | 无限制 |
| `1mo`, `3mo` | 月级 | 无限制 |

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rate_limit_delay` | 1.0 | 请求间隔时间（秒） |
| `max_retries` | 3 | 最大重试次数 |

## 返回数据格式

成功获取的数据为 pandas DataFrame，包含以下列：

| 列名 | 说明 |
|------|------|
| `Open` | 开盘价 |
| `High` | 最高价 |
| `Low` | 最低价 |
| `Close` | 收盘价 |
| `Volume` | 成交量 |
| `Dividends` | 股息 |
| `Stock Splits` | 股票分割 |
| `Range` | 价格区间 (High - Low) |
| `Change` | 价格变化 (Close - Open) |
| `Change_Pct` | 价格变化百分比 |

## 使用模板变量

在实际应用中，可以使用模板变量来动态设置参数：

```python
# 模板变量示例
ticker_name = "{{TICKER_NAME}}"      # 将被替换为实际股票代码
start_date = "{{START_DATE}}"        # 将被替换为实际开始日期
end_date = "{{END_DATE}}"            # 将被替换为实际结束日期
time_interval = "{{TIME_INTERVAL}}"  # 将被替换为实际时间间隔

# 在运行时替换模板变量
if "{{" in ticker_name:
    ticker_name = "AAPL"  # 实际值
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    time_interval = "1d"

# 获取数据
data = fetch_historical_data(ticker_name, start_date, end_date, time_interval)
```

## 错误处理

### 常见错误类型

1. **频率限制错误**: "Too Many Requests. Rate limited."
   - 自动重试机制会处理此类错误
   - 建议增加 `rate_limit_delay` 参数

2. **无效股票代码**: 返回空数据或异常
   - 检查股票代码是否正确
   - 确认股票在指定日期范围内有交易

3. **日期格式错误**: "日期格式错误，请使用YYYY-MM-DD格式"
   - 确保日期格式为 "YYYY-MM-DD"
   - 开始日期必须早于结束日期

4. **无效时间间隔**: "无效的时间间隔"
   - 检查时间间隔是否在支持的列表中

### 错误处理示例

```python
fetcher = YFinanceDataFetcher(rate_limit_delay=3.0, max_retries=2)

try:
    data = fetcher.get_historical_data("AAPL", "2024-01-01", "2024-01-31", "1d")
    
    if data is not None:
        print("数据获取成功")
        # 处理数据
    else:
        print("数据获取失败，请检查参数")
        
except Exception as e:
    print(f"发生未预期的错误: {e}")
```

## 最佳实践

### 1. 频率控制

```python
# 生产环境建议使用较长的延迟
fetcher = YFinanceDataFetcher(
    rate_limit_delay=5.0,  # 5秒延迟
    max_retries=2
)
```

### 2. 批量获取数据

```python
tickers = ["AAPL", "GOOGL", "MSFT"]
all_data = {}

fetcher = YFinanceDataFetcher(rate_limit_delay=3.0)

for ticker in tickers:
    print(f"正在获取 {ticker} 数据...")
    data = fetcher.get_historical_data(
        ticker, "2024-01-01", "2024-01-31", "1d"
    )
    if data is not None:
        all_data[ticker] = data
    
    # 额外延迟以避免频率限制
    time.sleep(2)
```

### 3. 数据缓存

```python
import pickle
import os
from datetime import datetime

def get_cached_data(ticker, start_date, end_date, interval):
    """获取缓存数据或从API获取新数据"""
    cache_file = f"cache_{ticker}_{start_date}_{end_date}_{interval}.pkl"
    
    # 检查缓存文件是否存在且不超过1天
    if os.path.exists(cache_file):
        file_time = os.path.getmtime(cache_file)
        if (datetime.now().timestamp() - file_time) < 86400:  # 24小时
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # 获取新数据
    fetcher = YFinanceDataFetcher(rate_limit_delay=3.0)
    data = fetcher.get_historical_data(ticker, start_date, end_date, interval)
    
    # 缓存数据
    if data is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    return data
```

## 运行示例

查看完整的使用示例：

```bash
python spider/yfinance_example.py
```

## 注意事项

1. **API限制**: yfinance 有严格的频率限制，建议在请求间添加适当延迟
2. **数据可用性**: 不同时间间隔的数据有不同的保留期限
3. **市场时间**: 只有在市场开放时间才有实时数据
4. **网络连接**: 确保网络连接稳定，API调用需要访问外部服务
5. **数据准确性**: 数据仅供参考，投资决策请谨慎

## 故障排除

### 问题：频繁遇到频率限制
**解决方案**:
- 增加 `rate_limit_delay` 到 5-10 秒
- 减少 `max_retries` 次数
- 考虑使用数据缓存

### 问题：获取不到数据
**解决方案**:
- 检查股票代码是否正确
- 确认日期范围内有交易数据
- 检查网络连接
- 尝试使用不同的时间间隔

### 问题：数据不完整
**解决方案**:
- 检查日期范围是否合理
- 确认时间间隔与数据保留期匹配
- 尝试分段获取数据

## 技术支持

如有问题，请参考：
- [yfinance 官方文档](https://ranaroussi.github.io/yfinance/reference/index.html)
- [pandas 文档](https://pandas.pydata.org/docs/)