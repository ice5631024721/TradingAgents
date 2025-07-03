#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAM策略回测执行脚本

本脚本用于执行波动率自适应动量策略的完整回测流程，
并生成详细的分析报告。
"""

import sys
import os
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vam_strategy import VAMStrategy

def generate_detailed_report(strategy, metrics):
    """
    生成详细的策略分析报告
    """
    report = []
    report.append("# 波动率自适应动量策略 (VAM) 回测报告")
    report.append("\n" + "=" * 80)
    report.append(f"\n## 策略概述")
    report.append(f"\n**策略名称**: 波动率自适应动量策略 (Volatility-Adaptive Momentum)")
    report.append(f"**交易标的**: {strategy.symbol}")
    report.append(f"**数据周期**: {strategy.period}")
    report.append(f"**回测天数**: {strategy.lookback_days}")
    report.append(f"**初始资金**: ${strategy.initial_capital:,.2f}")
    report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append(f"\n## 策略逻辑")
    report.append(f"\n### 核心触发条件")
    report.append(f"\n1. **动量确认**: 连续{strategy.momentum_periods}个周期收盘价高于{strategy.ma_period}周期均线，且MACD柱状线持续放大")
    report.append(f"2. **波动率过滤**: 当前ATR > 历史{strategy.atr_percentile}%分位数时触发策略")
    report.append(f"3. **量价背离修正**: 股价创新高但成交量低于前{strategy.volume_periods}周期均值时暂缓开仓")
    
    report.append(f"\n### 风险控制")
    report.append(f"\n- **止损比例**: {strategy.stop_loss*100:.1f}%")
    report.append(f"- **止盈比例**: {strategy.take_profit*100:.1f}%")
    report.append(f"- **仓位大小**: {strategy.position_size*100:.1f}%")
    
    report.append(f"\n## 回测结果")
    report.append(f"\n### 核心性能指标")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if '(%)' in key:
                report.append(f"- **{key}**: {value}%")
            elif key == '最终资产':
                report.append(f"- **{key}**: ${value:,.2f}")
            else:
                report.append(f"- **{key}**: {value}")
        else:
            report.append(f"- **{key}**: {value}")
    
    # 策略评估
    report.append(f"\n### 策略评估")
    
    # 收益率评估
    if metrics['年化收益率(%)'] > 15:
        return_assessment = "优秀"
    elif metrics['年化收益率(%)'] > 8:
        return_assessment = "良好"
    elif metrics['年化收益率(%)'] > 0:
        return_assessment = "一般"
    else:
        return_assessment = "较差"
    
    # 风险评估
    if abs(metrics['最大回撤(%)']) < 5:
        risk_assessment = "低风险"
    elif abs(metrics['最大回撤(%)']) < 15:
        risk_assessment = "中等风险"
    else:
        risk_assessment = "高风险"
    
    # 夏普比率评估
    if metrics['夏普比率'] > 1.5:
        sharpe_assessment = "优秀"
    elif metrics['夏普比率'] > 1.0:
        sharpe_assessment = "良好"
    elif metrics['夏普比率'] > 0.5:
        sharpe_assessment = "一般"
    else:
        sharpe_assessment = "较差"
    
    report.append(f"\n- **收益表现**: {return_assessment} (年化收益率 {metrics['年化收益率(%)']}%)")
    report.append(f"- **风险控制**: {risk_assessment} (最大回撤 {metrics['最大回撤(%)']}%)")
    report.append(f"- **风险调整收益**: {sharpe_assessment} (夏普比率 {metrics['夏普比率']})")
    report.append(f"- **相对表现**: {'跑赢基准' if metrics['超额收益(%)'] > 0 else '跑输基准'} (超额收益 {metrics['超额收益(%)']}%)")
    
    # 交易行为分析
    report.append(f"\n### 交易行为分析")
    report.append(f"\n- **交易频率**: {metrics['交易次数']} 次交易")
    report.append(f"- **胜率**: {metrics['胜率(%)']}%")
    report.append(f"- **最大连续亏损**: {metrics['最大连续亏损次数']} 次")
    
    if strategy.signals is not None:
        # 信号分析
        buy_signals = len(strategy.signals[strategy.signals['Signal'] == 1])
        sell_signals = len(strategy.signals[strategy.signals['Signal'] == -1])
        total_periods = len(strategy.signals)
        signal_rate = (buy_signals + sell_signals) / total_periods * 100
        
        report.append(f"- **信号生成率**: {signal_rate:.2f}% ({buy_signals + sell_signals}/{total_periods})")
        report.append(f"- **买入信号**: {buy_signals} 次")
        report.append(f"- **卖出信号**: {sell_signals} 次")
    
    # 策略优缺点分析
    report.append(f"\n## 策略分析")
    
    report.append(f"\n### 策略优势")
    advantages = []
    
    if metrics['夏普比率'] > 1.0:
        advantages.append("风险调整后收益表现良好")
    if metrics['胜率(%)'] > 50:
        advantages.append(f"胜率较高({metrics['胜率(%)']}%)，交易成功率可观")
    if metrics['超额收益(%)'] > 0:
        advantages.append(f"相对基准有超额收益({metrics['超额收益(%)']}%)")
    if abs(metrics['最大回撤(%)']) < 10:
        advantages.append("回撤控制较好，风险可控")
    
    if not advantages:
        advantages.append("策略在当前市场环境下表现一般")
    
    for i, advantage in enumerate(advantages, 1):
        report.append(f"{i}. {advantage}")
    
    report.append(f"\n### 策略劣势")
    disadvantages = []
    
    if metrics['年化收益率(%)'] < 5:
        disadvantages.append("年化收益率偏低，可能不足以覆盖通胀")
    if abs(metrics['最大回撤(%)']) > 15:
        disadvantages.append("最大回撤较大，风险偏高")
    if metrics['胜率(%)'] < 40:
        disadvantages.append("胜率偏低，需要优化信号质量")
    if metrics['交易次数'] < 10:
        disadvantages.append("交易频率较低，可能错过市场机会")
    if metrics['夏普比率'] < 0.5:
        disadvantages.append("夏普比率偏低，风险调整后收益不佳")
    
    if not disadvantages:
        disadvantages.append("策略整体表现良好，暂无明显劣势")
    
    for i, disadvantage in enumerate(disadvantages, 1):
        report.append(f"{i}. {disadvantage}")
    
    # 改进建议
    report.append(f"\n### 改进建议")
    suggestions = []
    
    if metrics['胜率(%)'] < 50:
        suggestions.append("优化信号过滤条件，提高信号质量")
    if abs(metrics['最大回撤(%)']) > 10:
        suggestions.append("加强风险控制，考虑动态调整止损位")
    if metrics['交易次数'] < 20:
        suggestions.append("适当放宽触发条件，增加交易机会")
    if metrics['夏普比率'] < 1.0:
        suggestions.append("优化仓位管理，提高风险调整后收益")
    
    suggestions.extend([
        "考虑加入更多市场状态判断指标",
        "测试不同参数组合的敏感性分析",
        "在不同市场环境下验证策略稳健性",
        "考虑加入机器学习方法优化信号生成"
    ])
    
    for i, suggestion in enumerate(suggestions, 1):
        report.append(f"{i}. {suggestion}")
    
    # 结论
    report.append(f"\n## 总结")
    
    if metrics['年化收益率(%)'] > 10 and abs(metrics['最大回撤(%)']) < 15 and metrics['夏普比率'] > 1.0:
        conclusion = "该VAM策略在回测期间表现优秀，具有良好的风险收益特征，建议进一步优化后实盘应用。"
    elif metrics['年化收益率(%)'] > 5 and metrics['超额收益(%)'] > 0:
        conclusion = "该VAM策略表现良好，相对基准有一定优势，但仍有优化空间，建议进一步测试和改进。"
    else:
        conclusion = "该VAM策略在当前参数设置下表现一般，需要进一步优化策略逻辑和参数配置。"
    
    report.append(f"\n{conclusion}")
    
    report.append(f"\n**注意**: 本回测结果基于历史数据，不代表未来表现。实际交易中需要考虑滑点、手续费、流动性等因素。")
    
    return "\n".join(report)

def save_report(report_content, filename):
    """
    保存报告到文件
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n详细报告已保存至: {filename}")
    except Exception as e:
        print(f"保存报告失败: {e}")

def main():
    """
    主函数：执行VAM策略回测
    """
    print("开始执行VAM策略回测...")
    print("=" * 80)
    
    try:
        # 创建策略实例
        print("\n正在初始化策略...")
        strategy = VAMStrategy(
            symbol='SPY',  # 可以改为其他标的，如'AAPL', 'TSLA'等
            period='5m',   # 5分钟K线
            lookback_days=30  # 回测30天
        )
        
        # 运行策略
        print("\n开始执行策略回测...")
        metrics = strategy.run_strategy()
        
        # 生成详细报告
        print("\n正在生成详细分析报告...")
        report_content = generate_detailed_report(strategy, metrics)
        
        # 保存报告
        report_filename = '/test_str/vam/VAM_Strategy_Report.md'
        save_report(report_content, report_filename)
        
        # 输出关键结果
        print("\n" + "=" * 80)
        print("VAM策略回测完成！")
        print("=" * 80)
        
        print(f"\n📊 关键指标:")
        print(f"   年化收益率: {metrics['年化收益率(%)']}%")
        print(f"   最大回撤: {metrics['最大回撤(%)']}%")
        print(f"   夏普比率: {metrics['夏普比率']}")
        print(f"   胜率: {metrics['胜率(%)']}%")
        print(f"   超额收益: {metrics['超额收益(%)']}%")
        
        print(f"\n📁 输出文件:")
        print(f"   策略图表: /Users/lingxiao/PycharmProjects/TradingAgents/test_str/vam_strategy_results.png")
        print(f"   详细报告: {report_filename}")
        
        # 简单的策略评估
        if metrics['年化收益率(%)'] > 10 and abs(metrics['最大回撤(%)']) < 15:
            print(f"\n✅ 策略评估: 表现优秀")
        elif metrics['超额收益(%)'] > 0:
            print(f"\n⚠️  策略评估: 表现良好，有优化空间")
        else:
            print(f"\n❌ 策略评估: 需要进一步优化")
            
        return True
        
    except Exception as e:
        print(f"\n❌ 策略执行失败: {str(e)}")
        print(f"\n错误详情:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 VAM策略回测成功完成！")
    else:
        print("\n💥 VAM策略回测执行失败，请检查错误信息。")
    
    print("\n" + "=" * 80)