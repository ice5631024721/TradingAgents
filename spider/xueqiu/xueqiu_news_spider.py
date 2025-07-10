"""
雪球实时新闻爬虫

功能：
1. 获取雪球实时新闻数据
2. 支持翻页获取历史数据
3. 数据清洗和格式化
4. 结果保存为JSON文件

作者: 量化交易专家
创建时间: 2025-07-04
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import urlencode, urlparse, parse_qs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XueqiuLiveNewsCrawler:
    """
    雪球实时新闻爬虫类
    """

    def __init__(self, delay_range=(1, 3), max_retries=3):
        """
        初始化爬虫

        Args:
            delay_range: 请求间隔范围(秒)
            max_retries: 最大重试次数
        """
        self.base_url = "https://xueqiu.com/statuses/livenews/list.json"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.session = requests.Session()

        # 设置请求头，模拟真实浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://xueqiu.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest'
        })

        # 初始化cookies（可能需要先访问主页获取）
        self._init_session()

    def _init_session(self):
        """
        初始化会话，获取必要的cookies和token
        """
        try:
            # 先访问雪球主页获取基础cookies
            logger.info("正在初始化会话...")
            response = self.session.get('https://xueqiu.com/', timeout=15)
            logger.info(f"访问主页成功，状态码: {response.status_code}")

            # 访问新闻页面获取更多cookies
            news_page = self.session.get('https://xueqiu.com/today', timeout=15)
            logger.info(f"访问新闻页面成功，状态码: {news_page.status_code}")

            # 尝试访问API端点获取token
            try:
                # 先尝试访问一个简单的API来获取认证信息
                test_api = self.session.get('https://xueqiu.com/service/csrf', timeout=10)
                if test_api.status_code == 200:
                    csrf_data = test_api.json()
                    if 'token' in csrf_data:
                        self.session.headers['X-CSRF-Token'] = csrf_data['token']
                        logger.info("获取CSRF token成功")
            except Exception as e:
                logger.debug(f"获取CSRF token失败: {e}")

            # 设置必要的请求头
            self.session.headers.update({
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            })

            logger.info("会话初始化完成")

        except Exception as e:
            logger.warning(f"初始化会话失败: {e}")

    def _build_url(self, count: int = 15, max_id: Optional[str] = None) -> str:
        """
        构建请求URL

        Args:
            count: 获取新闻数量
            max_id: 翻页ID，不传则获取最新数据

        Returns:
            完整的请求URL
        """
        params = {
            'count': count
        }

        if max_id:
            params['max_id'] = max_id

        # 添加md5参数（如果需要的话）
        # 这个参数可能是动态生成的，暂时不添加

        url = f"{self.base_url}?{urlencode(params)}"
        return url

    def _make_request(self, url: str) -> Optional[Dict[str, Any]]:
        """
        发送HTTP请求

        Args:
            url: 请求URL

        Returns:
            响应数据或None
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"正在请求: {url} (尝试 {attempt + 1}/{self.max_retries})")

                response = self.session.get(url, timeout=15)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"请求成功，获取到数据")

                        # 调试：打印原始返回数据的结构
                        logger.info(f"原始返回数据键: {list(data.keys())}")
                        if 'list' in data:
                            logger.info(f"list字段长度: {len(data['list'])}")
                        if 'items' in data:
                            logger.info(f"items字段长度: {len(data['items'])}")
                        if 'data' in data:
                            logger.info(f"data字段类型: {type(data['data'])}")
                        if 'statuses' in data:
                            logger.info(f"statuses字段长度: {len(data['statuses'])}")

                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}")
                        logger.debug(f"响应内容: {response.text[:500]}")
                        return None

                elif response.status_code == 400:
                    logger.error(f"请求失败，状态码: {response.status_code}")
                    logger.error(f"响应内容: {response.text}")
                    # 400错误通常是参数问题，不需要重试
                    return None

                else:
                    logger.warning(f"请求失败，状态码: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常: {e}")

            # 重试前等待
            if attempt < self.max_retries - 1:
                delay = random.uniform(*self.delay_range)
                logger.info(f"等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)

        logger.error(f"请求失败，已达到最大重试次数")
        return None

    def _parse_news_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析单条新闻数据

        Args:
            item: 原始新闻数据

        Returns:
            格式化后的新闻数据
        """
        try:
            # 提取基本信息
            news_id = item.get('id', '')
            text = item.get('text', '')
            created_at = item.get('created_at', 0)

            # 转换时间戳为发布时间
            if created_at:
                if len(str(created_at)) == 13:  # 毫秒时间戳
                    created_at = created_at / 1000
                publish_time = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
            else:
                publish_time = ''

            # 获取详情页URL
            url_detail = ''
            # 优先使用target字段，这是雪球提供的正确URL
            target_url = item.get('target', '')
            if target_url:
                # 确保使用https协议
                url_detail = target_url.replace('http://', 'https://')
            elif news_id:
                # 备用方案：使用status_id构建URL
                status_id = item.get('status_id', news_id)
                url_detail = f'https://xueqiu.com/statuses/{status_id}'

            return {
                'id': news_id,
                'text': text,
                'publish_time': publish_time,
                'url_detail': url_detail
            }

        except Exception as e:
            logger.error(f"解析新闻数据失败: {e}")
            return {
                'id': item.get('id', ''),
                'text': str(item),
                'url_detail': '',
                'error': str(e)
            }

    def _scrape_news_from_webpage(self, count: int = 15) -> Dict[str, Any]:
        """
        备用方案：从网页爬取新闻数据

        Args:
            count: 获取新闻数量

        Returns:
            包含新闻列表的字典
        """
        try:
            logger.info("使用备用方案：从网页爬取新闻数据")

            # 访问雪球今日页面
            response = self.session.get('https://xueqiu.com/today', timeout=15)

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'网页访问失败，状态码: {response.status_code}',
                    'news': [],
                    'total_count': 0
                }

            # 简单的HTML解析，提取新闻标题和链接
            import re
            html_content = response.text

            # 使用正则表达式提取新闻信息
            news_list = []

            # 提取新闻标题的模式
            title_patterns = [
                r'<a[^>]*href="([^"]*)">([^<]+)</a>',
                r'"title"\s*:\s*"([^"]+)"',
                r'"text"\s*:\s*"([^"]+)"'
            ]

            for pattern in title_patterns:
                matches = re.findall(pattern, html_content)
                for match in matches[:count]:
                    if isinstance(match, tuple) and len(match) == 2:
                        link, title = match
                        if len(title.strip()) > 10:  # 过滤太短的标题
                            url_detail = link if link.startswith('http') else f'https://xueqiu.com{link}'
                            news_list.append({
                                'id': f'web_{len(news_list)}',
                                'text': title.strip(),
                                'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'url_detail': url_detail,
                                'source': 'webpage_scraping'
                            })
                    elif isinstance(match, str) and len(match.strip()) > 10:
                        news_list.append({
                            'id': f'web_{len(news_list)}',
                            'text': match.strip(),
                            'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'url_detail': 'https://xueqiu.com/today',
                            'source': 'webpage_scraping'
                        })

                if len(news_list) >= count:
                    break

            # 去重
            seen_titles = set()
            unique_news = []
            for news in news_list:
                if news['title'] not in seen_titles:
                    seen_titles.add(news['title'])
                    unique_news.append(news)

            return {
                'success': True,
                'news': unique_news[:count],
                'total_count': len(unique_news[:count]),
                'source': 'webpage_scraping',
                'request_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"网页爬取失败: {e}")
            return {
                'success': False,
                'error': f'网页爬取失败: {str(e)}',
                'news': [],
                'total_count': 0
            }

    def get_news_by_time_range(self, start_time: str, end_time: str, max_count: int = 1000) -> Dict[str, Any]:
        """
        根据时间范围获取新闻

        Args:
            start_time: 开始时间，格式：'YYYY-MM-DD HH:MM:SS'
            end_time: 结束时间，格式：'YYYY-MM-DD HH:MM:SS'
            max_count: 最大获取数量，防止无限爬取

        Returns:
            包含新闻列表的字典
        """
        try:
            # 解析时间
            start_timestamp = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp()
            end_timestamp = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timestamp()

            if start_timestamp >= end_timestamp:
                return {
                    'success': False,
                    'error': '开始时间必须早于结束时间',
                    'news': [],
                    'total_count': 0
                }

        except ValueError as e:
            return {
                'success': False,
                'error': f'时间格式错误: {str(e)}',
                'news': [],
                'total_count': 0
            }

        logger.info(f"开始爬取时间范围内的新闻: {start_time} 到 {end_time}")

        all_news = []
        current_max_id = None
        page_count = 0
        max_pages = max_count // 15 + 1  # 估算最大页数
        should_continue = True  # 控制是否继续爬取的标志

        while (len(all_news) < max_count and
               page_count < max_pages and
               should_continue):
            page_count += 1
            logger.info(f"正在爬取第 {page_count} 页...")

            # 构建URL
            if page_count == 1:
                url = self._build_url(count=15)
            else:
                url = self._build_url(count=15, max_id=current_max_id)

            response_data = self._make_request(url)

            if not response_data or 'error_code' in response_data:
                logger.warning(f"第 {page_count} 页API访问失败")
                if response_data and 'error_code' in response_data:
                    logger.warning(f"API错误: {response_data.get('error_description')}")
                break

            # 解析API返回的新闻数据
            page_news = []

            # 尝试不同的数据字段
            statuses = response_data.get('statuses', []) or response_data.get('list', []) or response_data.get('items', [])

            # 如果data字段是字典且包含list
            if not statuses and 'data' in response_data:
                data_field = response_data['data']
                if isinstance(data_field, dict):
                    statuses = data_field.get('list', []) or data_field.get('items', [])
                elif isinstance(data_field, list):
                    statuses = data_field

            logger.info(f"第 {page_count} 页找到 {len(statuses)} 条原始新闻数据")

            # 解析新闻并过滤时间范围
            found_older_news = False
            found_newer_news = False
            valid_news_count = 0

            for item in statuses:
                # 先检查原始数据的时间戳
                created_at = item.get('created_at', 0)
                if created_at:
                    if len(str(created_at)) == 13:  # 毫秒时间戳
                        news_timestamp = created_at / 1000
                    else:
                        news_timestamp = created_at

                    # 检查新闻时间是否超出范围
                    if news_timestamp < start_timestamp:
                        # 新闻时间早于开始时间，说明已经超出范围，停止爬取
                        found_older_news = True
                        logger.info(f"发现早于开始时间的新闻: {datetime.fromtimestamp(news_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        break
                    elif news_timestamp > end_timestamp:
                        # 新闻时间晚于结束时间，跳过但继续处理后续新闻
                        found_newer_news = True
                        logger.debug(f"跳过晚于结束时间的新闻: {datetime.fromtimestamp(news_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        continue
                    else:
                        # 新闻时间在范围内，解析并添加到结果
                        parsed_item = self._parse_news_item(item)
                        page_news.append(parsed_item)
                        valid_news_count += 1
                else:
                    # 如果没有时间戳，记录警告但仍然解析添加（可能是特殊类型的新闻）
                    logger.warning(f"发现没有时间戳的新闻项: {item.get('id', 'unknown')}")
                    parsed_item = self._parse_news_item(item)
                    page_news.append(parsed_item)
                    valid_news_count += 1

            all_news.extend(page_news)
            logger.info(f"第 {page_count} 页筛选出 {len(page_news)} 条符合时间范围的新闻")

            # 判断是否应该停止爬取
            if found_older_news:
                logger.info("发现早于开始时间的新闻，停止爬取")
                should_continue = False
                break

            # 如果当前页没有找到任何有效新闻，可能已经超出时间范围
            if valid_news_count == 0 and len(statuses) > 0:
                logger.info("当前页没有找到符合时间范围的新闻，可能已超出范围")
                # 继续爬取一页以确认，但如果连续两页都没有有效新闻则停止
                if hasattr(self, '_consecutive_empty_pages'):
                    self._consecutive_empty_pages += 1
                    if self._consecutive_empty_pages >= 2:
                        logger.info("连续两页没有有效新闻，停止爬取")
                        should_continue = False
                        break
                else:
                    self._consecutive_empty_pages = 1
            else:
                # 重置连续空页计数器
                self._consecutive_empty_pages = 0

            # 获取下一页ID
            current_max_id = response_data.get('next_max_id') or response_data.get('nextMaxId')
            if not current_max_id:
                logger.info("没有更多数据，停止爬取")
                should_continue = False
                break

            # 页面间延迟
            if should_continue and page_count < max_pages:
                delay = random.uniform(*self.delay_range)
                logger.info(f"等待 {delay:.1f} 秒后继续...")
                time.sleep(delay)

        return {
            'success': True,
            'news': all_news,
            'total_count': len(all_news),
            'pages_crawled': page_count,
            'start_time': start_time,
            'end_time': end_time,
            'source': 'api',
            'request_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_latest_news(self, count: int = 15) -> Dict[str, Any]:
        """
        获取最新新闻（保留兼容性）

        Args:
            count: 获取新闻数量

        Returns:
            包含新闻列表的字典
        """
        # 获取当前时间作为结束时间
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 获取24小时前作为开始时间
        start_time = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')

        result = self.get_news_by_time_range(start_time, end_time, count)

        # 只返回指定数量的新闻
        if result['success'] and len(result['news']) > count:
            result['news'] = result['news'][:count]
            result['total_count'] = len(result['news'])

        return result

    def save_to_file(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        保存数据到JSON文件

        Args:
            data: 要保存的数据
            filename: 文件名，不指定则自动生成

        Returns:
            保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"xueqiu_livenews_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"数据已保存到文件: {filename}")
            return filename

        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return ''


def main():
    # 初始化爬虫
    crawler = XueqiuLiveNewsCrawler(delay_range=(2, 4), max_retries=3)
    # 获取过去6小时的新闻
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')

    print(f"时间范围: {start_time} 到 {end_time}")

    time_range_result = crawler.get_news_by_time_range(start_time, end_time, max_count=50)

    if time_range_result['success']:
        print(f"✓ 成功获取 {time_range_result['total_count']} 条时间范围内的新闻")
        print(f"✓ 爬取了 {time_range_result['pages_crawled']} 页")
        # 保存结果
        filename = crawler.save_to_file(time_range_result, "test_time_range_news.json")
        print(f"✓ 结果已保存到: {filename}")

    else:
        print(f"✗ 按时间范围获取新闻失败: {time_range_result['error']}")

if __name__ == "__main__":
    main()