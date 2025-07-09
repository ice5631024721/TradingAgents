#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新浪财经滚动新闻爬虫
使用crawl4ai框架爬取 https://finance.sina.com.cn/roll/ 的财经新闻数据
支持时间范围过滤功能
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    print("请安装 crawl4ai: pip install crawl4ai")
    raise

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("请安装 beautifulsoup4: pip install beautifulsoup4")
    raise

class SinaFinanceCrawler:
    """新浪财经滚动新闻爬虫"""
    
    def __init__(self, headless: bool = True):
        """初始化爬虫"""
        self.base_url = "https://finance.sina.com.cn/roll/"
        self.headless = headless
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def build_url(self, page: int = 1, num: int = 50) -> str:
        """构建新浪财经滚动新闻URL"""
        return f"{self.base_url}#pageid=384&lid=2672&k=&num={num}&page={page}"
    
    async def crawl_news_by_time_range(self, start_time: str, end_time: str, max_pages: int = 10) -> List[Dict]:
        """
        根据时间范围爬取新闻
        
        Args:
            start_time: 开始时间，格式: 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD'
            end_time: 结束时间，格式: 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD'
            max_pages: 最大爬取页数
            
        Returns:
            符合时间范围的新闻列表
        """
        print(f"开始爬取新浪财经新闻，时间范围: {start_time} 到 {end_time}")
        
        # 解析时间
        start_dt = self.parse_time(start_time)
        end_dt = self.parse_time(end_time)
        
        if not start_dt or not end_dt:
            raise ValueError("时间格式错误，请使用 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS' 格式")
        
        if start_dt > end_dt:
            raise ValueError("开始时间不能晚于结束时间")
        
        print(f"解析后的时间范围: {start_dt} 到 {end_dt}")
        
        all_news = []
        found_old_news = False
        
        for page in range(1, max_pages + 1):
            print(f"\n正在爬取第 {page} 页...")
            
            page_news = await self.crawl_single_page(page)
            if not page_news:
                print(f"第 {page} 页没有获取到新闻，停止爬取")
                break
            
            # 过滤时间范围内的新闻
            filtered_news = []
            for news in page_news:
                news_time = self.parse_news_time(news.get('time', ''))
                if news_time:
                    if start_dt <= news_time <= end_dt:
                        filtered_news.append(news)
                        print(f"  ✓ 符合时间范围: [{news_time}] {news['title'][:50]}...")
                    elif news_time < start_dt:
                        print(f"  ✗ 新闻时间早于开始时间: [{news_time}] {news['title'][:50]}...")
                        found_old_news = True
                    else:
                        print(f"  ✗ 新闻时间晚于结束时间: [{news_time}] {news['title'][:50]}...")
                else:
                    print(f"  ? 无法解析时间: {news.get('time', 'N/A')} - {news['title'][:50]}...")
            
            all_news.extend(filtered_news)
            print(f"第 {page} 页找到 {len(filtered_news)} 条符合条件的新闻")
            
            # 如果发现了早于开始时间的新闻，说明后续页面的新闻会更早，可以停止爬取
            if found_old_news and len(filtered_news) == 0:
                print("发现早于开始时间的新闻且当前页无符合条件新闻，停止爬取")
                break
            
            # 避免请求过快
            await asyncio.sleep(2)
        
        print(f"\n总共找到 {len(all_news)} 条符合时间范围的新闻")
        return all_news
    
    async def crawl_single_page(self, page: int = 1, num: int = 50) -> List[Dict]:
        """爬取单页新闻 - 使用crawl4ai框架"""
        url = self.build_url(page, num)
        print(f"爬取URL: {url}")
        
        try:
            # 根据crawl4ai文档，使用简化的配置
            async with AsyncWebCrawler(verbose=True) as crawler:
                # 使用基本的爬取配置
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    bypass_cache=True,
                    page_timeout=30000,
                    delay_before_return_html=3000,
                    js_code=[
                        "window.scrollTo(0, document.body.scrollHeight/2);",
                        "await new Promise(resolve => setTimeout(resolve, 2000));"
                    ]
                )
                
                if result.success and result.html:
                    print(f"crawl4ai爬取成功，HTML长度: {len(result.html)}")
                    
                    # 保存调试HTML
                    if page == 1:
                        debug_file = f'/Users/yangzhuo/project/pythonProject/TradingAgents/spider/debug_sina_finance_{int(time.time())}.html'
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(result.html)
                        print(f"调试HTML已保存到: {debug_file}")
                    
                    # 提取新闻
                    news_list = self.extract_news_from_html(result.html)
                    return news_list
                else:
                    print(f"页面爬取失败: {getattr(result, 'error_message', '未知错误')}")
                    return []
                    
        except Exception as e:
            print(f"爬取第 {page} 页时发生错误: {str(e)}")
            return []
    
    def extract_news_from_html(self, html: str) -> List[Dict]:
        """从HTML中提取新闻数据"""
        news_list = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            print(f"HTML解析成功，页面标题: {soup.title.string if soup.title else '无标题'}")
            
            # 专门针对新浪财经的新闻列表结构
            # 查找包含新闻的li元素
            news_items = soup.select('li')
            
            for li in news_items:
                try:
                    # 查找标题和链接
                    title_element = li.select_one('span.c_tit a')
                    if not title_element:
                        continue
                    
                    title = title_element.get_text(strip=True)
                    link = title_element.get('href', '')
                    
                    # 提取真实的新闻链接
                    real_link = self.extract_real_link(link)
                    
                    # 查找时间
                    time_element = li.select_one('span.c_time')
                    time_str = ''
                    if time_element:
                        # 优先使用页面显示的时间文本（如：07-10 00:23）
                        time_str = time_element.get_text(strip=True)
                    
                    if not time_str:
                        time_str = datetime.now().strftime('%H:%M')
                    
                    if title and len(title) > 10:  # 确保标题有意义
                         # 解析时间
                         parsed_time = self.parse_news_time(time_str)
                         
                         news_item = {
                             'id': f'sina_finance_{int(time.time())}_{len(news_list)}',
                             'title': title,
                             'time': parsed_time.strftime('%Y-%m-%d %H:%M:%S') if parsed_time else None,
                             'url': real_link if real_link else self.base_url
                         }
                         news_list.append(news_item)
                        
                except Exception as e:
                    print(f"处理单个新闻项时出错: {str(e)}")
                    continue
            
            # 如果没有找到结构化新闻，尝试文本模式
            if len(news_list) < 5:
                print(f"结构化提取的新闻数量较少({len(news_list)})，尝试文本模式补充")
                text_news = self.extract_news_by_text_patterns(html)
                news_list.extend(text_news)
            
            # 去重
            news_list = self.deduplicate_news(news_list)
            
            print(f"成功提取 {len(news_list)} 条新闻")
            return news_list
            
        except Exception as e:
            print(f"解析HTML时发生错误: {str(e)}")
            return []
    
    def extract_real_link(self, link: str) -> str:
        """提取真实的新闻链接"""
        if not link:
            return ''
        
        try:
            # 如果是重定向链接，提取真实URL
            if 'cj.sina.cn/article/norm_detail?url=' in link:
                # 解析URL参数
                from urllib.parse import urlparse, parse_qs, unquote
                parsed = urlparse(link)
                query_params = parse_qs(parsed.query)
                if 'url' in query_params:
                    real_url = unquote(query_params['url'][0])
                    return real_url
            
            # 如果已经是完整的finance.sina.com.cn链接
            if link.startswith('https://finance.sina.com.cn/'):
                return link
            
            # 如果是相对链接，补全域名
            if link.startswith('/'):
                return 'https://finance.sina.com.cn' + link
            
            return link
            
        except Exception as e:
            print(f"提取真实链接时出错: {str(e)}")
            return link
    

    
    def extract_time_from_element(self, element, text: str) -> str:
        """从元素中提取时间"""
        time_str = None
        
        # 尝试从属性中获取时间
        for attr in ['data-time', 'data-timestamp', 'time', 'datetime', 'data-date']:
            if element.get(attr):
                time_str = element.get(attr)
                break
        
        if not time_str:
            # 尝试从子元素中获取时间
            time_element = element.find(['time', 'span'], class_=re.compile(r'time|date', re.I))
            if time_element:
                time_str = time_element.get_text(strip=True)
        
        if not time_str:
            # 尝试从文本中提取时间模式
            time_patterns = [
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})',
                r'(\d{2}-\d{2} \d{2}:\d{2})',
                r'(\d{2}:\d{2})',
                r'(\d{1,2}月\d{1,2}日 \d{2}:\d{2})',
                r'(\d{2}/\d{2} \d{2}:\d{2})',
                r'(今天 \d{2}:\d{2})',
                r'(昨天 \d{2}:\d{2})',
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{1,2}月\d{1,2}日)'
            ]
            for pattern in time_patterns:
                match = re.search(pattern, text)
                if match:
                    time_str = match.group(1)
                    break
        
        if not time_str:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return time_str
    
    def extract_title_and_content(self, element, text: str) -> tuple:
        """提取标题和内容"""
        title = text[:100] if len(text) > 100 else text
        content = text
        
        # 尝试从结构中提取更详细的信息
        title_element = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', 'a'])
        if title_element:
            title = title_element.get_text(strip=True)
            # 如果找到了标题元素，尝试获取剩余内容作为正文
            remaining_text = text.replace(title, '').strip()
            if remaining_text:
                content = remaining_text
        
        return title, content
    
    def extract_news_by_text_patterns(self, html: str) -> List[Dict]:
        """通过文本模式提取新闻"""
        news_list = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()
            
            # 按行分割文本
            lines = text_content.split('\n')
            potential_news = []
            
            for line in lines:
                line = line.strip()
                # 筛选可能的新闻行
                if (20 <= len(line) <= 300 and 
                    not any(keyword in line.lower() for keyword in ['登录', '注册', '首页', '导航', '菜单', 'copyright', '版权', '更多', '加载']) and
                    (re.search(r'\d{2}:\d{2}', line) or  # 包含时间
                     any(keyword in line for keyword in ['股市', '股票', '基金', '期货', '债券', '外汇', '黄金', '原油', '经济', '财经', '金融', '银行', '保险', '证券', '投资', '市场', '交易', '涨跌', '收盘', '开盘']))):
                    potential_news.append(line)
            
            # 转换为新闻项
            for i, news_text in enumerate(potential_news[:20]):
                time_match = re.search(r'(\d{2}:\d{2})', news_text)
                time_str = time_match.group(1) if time_match else datetime.now().strftime('%H:%M')
                
                # 解析时间
                parsed_time = self.parse_news_time(time_str)
                
                news_list.append({
                    'id': f'sina_finance_text_{int(time.time())}_{i}',
                    'title': news_text[:80] + '...' if len(news_text) > 80 else news_text,
                    'time': parsed_time.strftime('%Y-%m-%d %H:%M:%S') if parsed_time else None,
                    'url': self.base_url
                })
            
            return news_list
            
        except Exception as e:
            print(f"文本模式提取时发生错误: {str(e)}")
            return []
    
    def deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """去重新闻"""
        seen_content = set()
        unique_news = []
        
        for news in news_list:
            content_hash = hash(news['title'][:50])  # 使用标题前50字符的哈希值去重
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_news.append(news)
        
        return unique_news
    
    def parse_time(self, time_str: str) -> Optional[datetime]:
        """解析时间字符串"""
        try:
            # 尝试多种时间格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y/%m/%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            
            # 如果只有日期，添加时间
            if len(time_str) == 10 and '-' in time_str:
                return datetime.strptime(time_str + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
            
            return None
            
        except Exception as e:
            print(f"解析时间失败: {time_str}, 错误: {str(e)}")
            return None
    
    def parse_news_time(self, time_str: str) -> Optional[datetime]:
        """解析新闻时间"""
        try:
            # 处理新浪财经的时间格式："07-10 00:23"
            sina_time_match = re.match(r'^(\d{2})-(\d{2}) (\d{2}):(\d{2})$', time_str.strip())
            if sina_time_match:
                month = int(sina_time_match.group(1))
                day = int(sina_time_match.group(2))
                hour = int(sina_time_match.group(3))
                minute = int(sina_time_match.group(4))
                year = datetime.now().year
                return datetime(year, month, day, hour, minute)
            
            # 处理相对时间
            if '今天' in time_str:
                time_part = re.search(r'(\d{2}:\d{2})', time_str)
                if time_part:
                    today = datetime.now().date()
                    time_obj = datetime.strptime(time_part.group(1), '%H:%M').time()
                    return datetime.combine(today, time_obj)
            
            if '昨天' in time_str:
                time_part = re.search(r'(\d{2}:\d{2})', time_str)
                if time_part:
                    yesterday = datetime.now().date() - timedelta(days=1)
                    time_obj = datetime.strptime(time_part.group(1), '%H:%M').time()
                    return datetime.combine(yesterday, time_obj)
            
            # 处理只有时间的情况（如 "00:59"）
            time_only_match = re.match(r'^(\d{2}:\d{2})$', time_str.strip())
            if time_only_match:
                today = datetime.now().date()
                time_obj = datetime.strptime(time_only_match.group(1), '%H:%M').time()
                return datetime.combine(today, time_obj)
            
            # 处理月日格式
            month_day_match = re.search(r'(\d{1,2})月(\d{1,2})日', time_str)
            if month_day_match:
                month = int(month_day_match.group(1))
                day = int(month_day_match.group(2))
                year = datetime.now().year
                
                time_part = re.search(r'(\d{2}:\d{2})', time_str)
                if time_part:
                    time_obj = datetime.strptime(time_part.group(1), '%H:%M').time()
                    return datetime.combine(datetime(year, month, day).date(), time_obj)
                else:
                    return datetime(year, month, day)
            
            # 使用通用解析
            return self.parse_time(time_str)
            
        except Exception as e:
            print(f"? 无法解析时间: {time_str} - 错误: {str(e)}")
            return None
    
    def save_news_to_file(self, news_list: List[Dict], filename: str = None) -> str:
        """保存新闻到文件"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'/Users/yangzhuo/project/pythonProject/TradingAgents/spider/sina_finance_news_{timestamp}.json'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_list, f, ensure_ascii=False, indent=2, default=str)
            print(f"新闻数据已保存到: {filename}")
            return filename
        except Exception as e:
            print(f"保存文件时发生错误: {str(e)}")
            return ""

async def main():
    """主函数 - 示例用法"""
    crawler = SinaFinanceCrawler(headless=True)
    
    # 示例1: 爬取今天的新闻
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n=== 示例1: 爬取今天({today})的新闻 ===")
    
    news_list = await crawler.crawl_news_by_time_range(
        start_time=f"{today} 00:00:00",
        end_time=f"{today} 23:59:59",
        max_pages=3
    )
    
    if news_list:
        print(f"\n成功获取 {len(news_list)} 条今天的新闻:")
        for i, news in enumerate(news_list[:5], 1):  # 显示前5条
            print(f"{i}. [{news['time']}] {news['title'][:60]}...")
        
        # 保存到文件
        filename = crawler.save_news_to_file(news_list)
        if filename:
            print(f"\n数据已保存到: {filename}")
    else:
        print("未能获取到新闻数据")
    
    # 示例2: 爬取最近3天的新闻
    print(f"\n\n=== 示例2: 爬取最近3天的新闻 ===")
    
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    news_list_3days = await crawler.crawl_news_by_time_range(
        start_time=f"{start_date} 00:00:00",
        end_time=f"{end_date} 23:59:59",
        max_pages=5
    )
    
    if news_list_3days:
        print(f"\n成功获取 {len(news_list_3days)} 条最近3天的新闻")
        
        # 按日期统计
        date_stats = {}
        for news in news_list_3days:
            news_time = crawler.parse_news_time(news.get('time', ''))
            if news_time:
                date_key = news_time.strftime('%Y-%m-%d')
                date_stats[date_key] = date_stats.get(date_key, 0) + 1
        
        print("\n按日期统计:")
        for date, count in sorted(date_stats.items()):
            print(f"  {date}: {count} 条新闻")
    
if __name__ == "__main__":
    asyncio.run(main())