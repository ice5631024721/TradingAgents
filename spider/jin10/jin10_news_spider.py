"""
金十数据快讯爬虫
爬取 https://flash-api.jin10.com/get_flash_list 接口数据
支持时间范围过滤和翻页功能
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode


class Jin10FlashCrawler:
    """金十数据快讯爬虫类"""

    def __init__(self, delay: float = 1.0, max_retries: int = 3):
        """
        初始化爬虫

        Args:
            delay: 请求间隔时间（秒）
            max_retries: 最大重试次数
        """
        self.base_url = "https://flash-api.jin10.com/get_flash_list"
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()

        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.jin10.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'x-app-id': 'bVBF4FyRTn5NJF5n',
            'x-version': '1.0.0'
        })

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """
        发送HTTP请求

        Args:
            params: 请求参数

        Returns:
            响应数据或None
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"发送请求 (尝试 {attempt + 1}/{self.max_retries}): {params}")

                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=30
                )

                self.logger.info(f"响应状态码: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(
                        f"成功获取数据，数据键: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
                    return data
                elif response.status_code == 502:
                    self.logger.warning(f"服务器错误 502，尝试 {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
                        continue
                else:
                    self.logger.error(f"请求失败，状态码: {response.status_code}")
                    self.logger.error(f"响应内容: {response.text[:500]}")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {e}")
                break

        return None

    def _parse_time(self, time_str: str) -> datetime:
        """
        解析时间字符串

        Args:
            time_str: 时间字符串，支持多种格式

        Returns:
            datetime对象
        """
        # 支持的时间格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d+%H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d+%H:%M',
            '%Y-%m-%d'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {time_str}")

    def _format_time_for_api(self, dt: datetime) -> str:
        """
        将datetime对象格式化为API需要的时间格式

        Args:
            dt: datetime对象

        Returns:
            格式化的时间字符串
        """
        return dt.strftime('%Y-%m-%d+%H:%M:%S')

    def _extract_flash_data(self, raw_data: Dict) -> List[Dict]:
        """
        从原始响应数据中提取快讯数据

        Args:
            raw_data: API原始响应数据

        Returns:
            快讯数据列表
        """
        # 根据搜索结果，数据可能在不同的字段中
        flash_list = []
        if 'data' in raw_data:
            flash_list = raw_data['data']
        elif 'list' in raw_data:
            flash_list = raw_data['list']
        elif 'items' in raw_data:
            flash_list = raw_data['items']
        elif isinstance(raw_data, list):
            flash_list = raw_data
        else:
            self.logger.warning(
                f"未知的数据结构: {list(raw_data.keys()) if isinstance(raw_data, dict) else type(raw_data)}")
            return []

        # 处理每个快讯项，提取链接信息
        processed_list = []
        for item in flash_list:
            processed_item = self._process_flash_item(item)
            processed_list.append(processed_item)

        return processed_list

    def _process_flash_item(self, item: Dict) -> Dict:
        """
        处理单个快讯项，提取链接信息

        Args:
            item: 原始快讯数据项

        Returns:
            处理后的快讯数据项
        """
        # 复制原始数据
        processed_item = item.copy()

        # 如果是重要快讯(important=1)，尝试提取链接
        if item.get('important') == 1:
            links = self._extract_links_from_item(item)
            if links:
                processed_item['links'] = links

        return processed_item

    def _extract_links_from_item(self, item: Dict) -> List[str]:
        """
        从快讯项中提取链接信息

        Args:
            item: 快讯数据项

        Returns:
            链接列表
        """
        links = []

        # 从remark字段中提取链接
        if 'remark' in item and isinstance(item['remark'], list):
            for remark_item in item['remark']:
                if isinstance(remark_item, dict):
                    # 检查url字段
                    if 'url' in remark_item:
                        url = remark_item['url']
                        if url and url.startswith('http'):
                            links.append(url)
                    # 检查link字段
                    elif 'link' in remark_item:
                        link = remark_item['link']
                        if link and link.startswith('http'):
                            links.append(link)

        # 从data字段中提取链接
        if 'data' in item and isinstance(item['data'], dict):
            data = item['data']
            # 检查source_link字段
            source_link = data.get('source_link')
            if source_link and source_link.startswith('http'):
                links.append(source_link)
            # 检查link字段
            link = data.get('link')
            if link and link.startswith('http'):
                links.append(link)

        return links

    def _get_earliest_time_from_data(self, flash_list: List[Dict]) -> Optional[str]:
        """
        从快讯数据中获取最早的时间，用于翻页

        Args:
            flash_list: 快讯数据列表

        Returns:
            最早的时间字符串或None
        """
        if not flash_list:
            return None

        # 尝试不同的时间字段名
        time_fields = ['time', 'created_at', 'publish_time', 'datetime']

        earliest_time = None
        for item in flash_list:
            for field in time_fields:
                if field in item and item[field]:
                    try:
                        # 如果是时间戳
                        if isinstance(item[field], (int, float)):
                            item_time = datetime.fromtimestamp(item[field])
                        else:
                            item_time = self._parse_time(str(item[field]))

                        if earliest_time is None or item_time < earliest_time:
                            earliest_time = item_time
                        break
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"解析时间失败: {item[field]}, 错误: {e}")
                        continue

        return self._format_time_for_api(earliest_time) if earliest_time else None

    def _is_in_time_range(self, item: Dict, start_time: datetime, end_time: datetime) -> bool:
        """
        检查快讯是否在指定时间范围内

        Args:
            item: 快讯数据项
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            是否在时间范围内
        """
        time_fields = ['time', 'created_at', 'publish_time', 'datetime']

        for field in time_fields:
            if field in item and item[field]:
                try:
                    # 如果是时间戳
                    if isinstance(item[field], (int, float)):
                        item_time = datetime.fromtimestamp(item[field])
                    else:
                        item_time = self._parse_time(str(item[field]))

                    return start_time <= item_time <= end_time
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"解析时间失败: {item[field]}, 错误: {e}")
                    continue

        self.logger.warning(f"无法找到有效的时间字段: {list(item.keys())}")
        return False

    def _parse_timestamp(self, timestamp: int) -> datetime:
        """
        解析时间戳

        Args:
            timestamp: 时间戳（秒或毫秒）

        Returns:
            datetime对象
        """
        # 如果是毫秒时间戳，转换为秒
        if timestamp > 10 ** 10:
            timestamp = timestamp / 1000

        return datetime.fromtimestamp(timestamp)

    def _extract_item_time(self, item: Dict) -> Optional[datetime]:
        """
        从快讯项目中提取时间

        Args:
            item: 快讯数据项

        Returns:
            datetime对象或None
        """
        time_fields = ['time', 'created_at', 'publish_time', 'datetime']

        for field in time_fields:
            if field in item and item[field]:
                try:
                    if isinstance(item[field], (int, float)):
                        return datetime.fromtimestamp(item[field])
                    else:
                        return self._parse_time(str(item[field]))
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"解析时间字段 {field} 失败: {item[field]}, 错误: {e}")
                    continue

        return None

    def get_flash_news_by_time_range(
            self,
            start_time: str,
            end_time: str,
            channel: str = "-8200",
            vip: str = "1",
            classify: str = "[19]",
            max_pages: int = 50
    ) -> List[Dict]:
        """
        根据时间范围获取快讯数据

        Args:
            start_time: 开始时间，格式如 '2024-01-01 00:00:00'
            end_time: 结束时间，格式如 '2024-01-01 23:59:59'
            channel: 频道参数，默认-8200
            vip: VIP参数，默认1
            classify: 分类参数，默认[19]表示数字货币相关
            max_pages: 最大翻页数

        Returns:
            符合时间范围的快讯数据列表
        """
        self.logger.info(f"开始爬取金十数据快讯，时间范围: {start_time} 到 {end_time}")

        # 解析时间
        try:
            start_dt = self._parse_time(start_time)
            end_dt = self._parse_time(end_time)
        except ValueError as e:
            self.logger.error(f"时间格式错误: {e}")
            return []

        if start_dt >= end_dt:
            self.logger.error("开始时间必须早于结束时间")
            return []

        all_flash_news = []
        page_count = 0
        consecutive_empty_pages = 0

        # 格式化时间为API需要的格式（YYYY-MM-DD）
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')

        # 使用翻页方式获取数据，因为API使用max_time参数进行翻页
        current_max_time = None
        should_stop_next = False  # 标记下一轮是否应该停止

        while page_count <= max_pages and not should_stop_next:
            self.logger.info(f"正在爬取第 {page_count} 页数据")

            # 构建请求参数
            params = {
                'channel': channel,
                'vip': vip,
                'classify': classify
            }

            # 添加max_time参数用于翻页
            if current_max_time:
                params['max_time'] = current_max_time

            # 发送请求
            raw_data = self._make_request(params)
            if raw_data is None:
                self.logger.error(f"第 {page_count} 页请求失败")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    break
                page_count += 1
                continue

            # 提取快讯数据
            flash_list = self._extract_flash_data(raw_data)
            self.logger.info(f"第 {page_count} 页获取到 {len(flash_list)} 条原始数据")

            if not flash_list:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    self.logger.warning("连续3页无数据，停止爬取")
                    break
                page_count += 1
                continue
            else:
                consecutive_empty_pages = 0

            # 过滤时间范围内的数据
            valid_news = []
            earliest_time = None

            for item in flash_list:
                # 获取当前项的时间
                item_time = self._extract_item_time(item)
                if item_time is None:
                    continue

                # 检查时间范围
                if start_dt <= item_time <= end_dt:
                    # 在时间范围内，添加到结果
                    valid_news.append(item)
                elif item_time < start_dt:
                    # 早于开始时间，标记下一轮停止
                    should_stop_next = True

                # 更新最早时间用于翻页
                if earliest_time is None or item_time < earliest_time:
                    earliest_time = item_time

            self.logger.info(f"第 {page_count} 页筛选出 {len(valid_news)} 条符合时间范围的数据")
            all_flash_news.extend(valid_news)

            # 设置下一页的max_time参数
            if earliest_time and not should_stop_next:
                current_max_time = self._format_time_for_api(earliest_time)
            elif should_stop_next:
                self.logger.info("发现早于开始时间的数据，下一轮将停止爬取")
            else:
                self.logger.info("无法获取下一页时间参数，停止爬取")
                break

            page_count += 1

            # 请求间隔
            if page_count <= max_pages and not should_stop_next:
                time.sleep(self.delay)

        self.logger.info(f"爬取完成，共获取 {len(all_flash_news)} 条符合条件的快讯数据")
        return all_flash_news

    def save_to_file(self, data: List[Dict], filename: str) -> None:
        """
        保存数据到文件

        Args:
            data: 要保存的数据
            filename: 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"数据已保存到 {filename}")
        except Exception as e:
            self.logger.error(f"保存文件失败: {e}")


def main():
    """主函数，用于测试"""
    crawler = Jin10FlashCrawler(delay=1.0)

    # 测试用例1: 获取最近1小时的数据
    print("\n=== 测试1: 获取最近1小时的快讯 ===")
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    news = crawler.get_flash_news_by_time_range(
        start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
        end_time=end_time.strftime('%Y-%m-%d %H:%M:%S'),
        max_pages=5
    )

    print(f"获取到 {len(news)} 条快讯")
    if news:
        crawler.save_to_file(news, 'jin10_flash_recent_1hour.json')


if __name__ == "__main__":
    main()