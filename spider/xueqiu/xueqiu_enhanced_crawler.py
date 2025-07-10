#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雪球网站增强版数据爬虫
支持代理设置、反爬机制和时间范围过滤
"""

import asyncio
import json
import re
import random
import time
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import List, Dict, Optional

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai import LLMExtractionStrategy, JsonCssExtractionStrategy
    from crawl4ai import PruningContentFilter, BM25ContentFilter
    from crawl4ai import DefaultMarkdownGenerator
except ImportError:
    print("请先安装 crawl4ai: pip install crawl4ai")
    exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装 beautifulsoup4: pip install beautifulsoup4")
    exit(1)

class XueqiuEnhancedCrawler:
    """雪球网站增强版爬虫"""
    
    def __init__(self, use_proxy: bool = False, proxy_list: Optional[List[str]] = None):
        """
        初始化爬虫
        
        Args:
            use_proxy: 是否使用代理
            proxy_list: 代理列表，格式如 ['http://proxy1:port', 'http://proxy2:port']
        """
        self.use_proxy = use_proxy
        self.proxy_list = proxy_list or []
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
    def format_url(self, query_name: str) -> str:
        """格式化雪球搜索URL"""
        encoded_query = quote(query_name)
        return f"https://xueqiu.com/k?q={encoded_query}#/timeline"
    
    def parse_time_string(self, time_str: str) -> Optional[datetime]:
        """解析时间字符串为datetime对象"""
        try:
            # 处理相对时间
            now = datetime.now()
            
            if '分钟前' in time_str:
                minutes = int(re.search(r'(\d+)分钟前', time_str).group(1))
                return now - timedelta(minutes=minutes)
            elif '小时前' in time_str:
                hours = int(re.search(r'(\d+)小时前', time_str).group(1))
                return now - timedelta(hours=hours)
            elif '天前' in time_str:
                days = int(re.search(r'(\d+)天前', time_str).group(1))
                return now - timedelta(days=days)
            elif '昨天' in time_str:
                return now - timedelta(days=1)
            elif '前天' in time_str:
                return now - timedelta(days=2)
            elif '刚刚' in time_str or '刚才' in time_str:
                return now
            else:
                # 尝试解析具体日期格式
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m-%d %H:%M', '%m月%d日']:
                    try:
                        return datetime.strptime(time_str, fmt)
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"解析时间失败: {time_str}, 错误: {e}")
            
        return None
    
    def format_time_to_standard(self, time_str: str) -> str:
        """将时间字符串格式化为标准格式 YYYY-MM-DD HH:MM:SS"""
        try:
            parsed_time = self.parse_time_string(time_str)
            if parsed_time:
                # 确保年份正确（如果是未来年份，调整为当前年份）
                current_year = datetime.now().year
                if parsed_time.year > current_year:
                    parsed_time = parsed_time.replace(year=current_year)
                return parsed_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # 如果解析失败，返回当前时间
                return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"格式化时间失败: {time_str}, 错误: {e}")
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def is_in_time_range(self, post_time: datetime, start_time: datetime, end_time: datetime) -> bool:
        """检查帖子时间是否在指定范围内"""
        return start_time <= post_time <= end_time
    
    def is_irrelevant_content(self, content: str) -> bool:
        """判断内容是否为页面无关内容（如版权信息、广告、登录提示等）"""
        if not content or len(content.strip()) < 10:
            return True
        
        # 首先检查是否包含有价值的投资内容关键词
        valuable_keywords = [
            '特斯拉', 'TSLA', '股票', '投资', '交付', '财报', '业绩', '分析', 
            '看多', '看空', '买入', '卖出', '持有', '涨', '跌', '行情', '市场',
            '美股', '盘前', '盘后', '预期', '万辆', '季度', '全球', '分析师',
            '马斯克', '销量', '竞争', '欧洲', '销售', '挑战', '下滑'
        ]
        
        # 如果包含有价值的关键词，则不过滤
        if any(keyword in content for keyword in valuable_keywords):
            return False
        
        # 定义无关内容的关键词模式（更严格的匹配）
        irrelevant_patterns = [
            # 版权和法律信息（完整匹配）
            r'^.*版权.*雪球.*$',
            r'^.*XUEQIU\.COM.*$',
            r'^.*京ICP.*$',
            r'^.*京公网安备.*$',
            r'^.*营业执照.*$',
            
            # 登录和安全提示（完整匹配）
            r'^.*账号安全等级低.*$',
            r'^.*系统检测到您的雪球账号.*$',
            r'^.*绑定手机号.*$',
            r'^.*扫一扫.*关注雪球.*$',
            
            # 功能按钮和导航（精确匹配）
            r'^(登录|注册|下载|分享|举报|收藏|关注|取消关注)$',
            r'^(发送验证码|重新发送|确定|取消|关闭)$',
            r'^(默认排序|最新讨论|讨论最多|综合)$',
            r'^(股票|组合|用户|搜索|首页)$',
            
            # 讨论限制提示
            r'^.*本帖讨论暂时受限.*$',
            r'^.*防诈骗.*$',
            
            # 长串的国家代码（多个连续的国家代码）
            r'.*\+\d{1,4}.*\+\d{1,4}.*\+\d{1,4}.*',
        ]
        
        # 检查是否匹配无关内容模式
        for pattern in irrelevant_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                return True
        
        # 检查是否主要由数字、符号和空格组成（放宽标准）
        clean_content = re.sub(r'[\s\d\+\-\(\)\[\]\{\}\|\\]', '', content)
        if len(clean_content) < len(content) * 0.1:  # 如果有意义字符少于10%（从30%降低到10%）
            return True
        
        # 检查是否包含过多重复字符（放宽标准）
        if len(set(content)) < len(content) * 0.05:  # 如果唯一字符少于5%（从10%降低到5%）
            return True
        
        return False
    
    def get_random_proxy(self) -> Optional[str]:
        """获取随机代理"""
        if self.use_proxy and self.proxy_list:
            return random.choice(self.proxy_list)
        return None
    
    def get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self.user_agents)
    
    async def crawl_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """带重试机制的爬取 - 使用crawl4ai最佳实践"""
        for attempt in range(max_retries):
            try:
                # 随机延迟
                await asyncio.sleep(random.uniform(2, 5))
                
                # 配置浏览器 - 使用crawl4ai的BrowserConfig
                browser_config = BrowserConfig(
                    headless=True,
                    browser_type="chromium",
                    user_agent=self.get_random_user_agent(),
                    viewport_width=1920,
                    viewport_height=1080,
                    accept_downloads=False,
                    java_script_enabled=True,
                    ignore_https_errors=True,
                    extra_args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor"
                    ]
                )
                
                # 添加代理配置
                proxy = self.get_random_proxy()
                if proxy:
                    browser_config.proxy = proxy
                    print(f"使用代理: {proxy}")
                
                # 配置爬取运行参数 - 使用crawl4ai的CrawlerRunConfig
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,  # 绕过缓存确保获取最新内容
                    
                    # JavaScript执行配置
                    js_code=[
                        # 第一步：基础页面准备
                        """
                        console.log('🚀 开始雪球网站智能爬取...');
                        
                        // 等待页面基础加载
                        await new Promise(resolve => {
                            if (document.readyState === 'complete') {
                                resolve();
                            } else {
                                window.addEventListener('load', resolve);
                            }
                        });
                        
                        console.log('✅ 页面基础加载完成');
                        """,
                        
                        # 第二步：处理弹窗和导航
                        """
                        console.log('🔧 处理弹窗和页面导航...');
                        
                        // 关闭可能的弹窗
                        const modalSelectors = [
                            '.modal__login', '.modal__security-alert', '.modal__confirm', 
                            '.modal__alert', '.widget__download-app', '.taichi__toast',
                            '[class*="modal"]', '[class*="popup"]', '[class*="dialog"]'
                        ];
                        
                        for (const selector of modalSelectors) {
                            const modals = document.querySelectorAll(selector);
                            modals.forEach(modal => {
                                if (modal && getComputedStyle(modal).display !== 'none') {
                                    const closeBtn = modal.querySelector('.close, .no-more, [class*="close"], [aria-label*="close"]');
                                    if (closeBtn) {
                                        closeBtn.click();
                                        console.log('❌ 关闭弹窗:', selector);
                                    }
                                }
                            });
                        }
                        
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        
                        // 确保在讨论页面
                        const timelineTab = document.querySelector('a[href="#/timeline"], [href*="timeline"]');
                        if (timelineTab && !timelineTab.classList.contains('active')) {
                            timelineTab.click();
                            console.log('📋 切换到讨论页面');
                            await new Promise(resolve => setTimeout(resolve, 3000));
                        }
                        
                        console.log('✅ 弹窗处理和导航完成');
                        """,
                        
                        # 第三步：等待动态内容加载
                        """
                        console.log('⏳ 等待动态内容加载...');
                        
                        // 智能等待内容加载
                        let contentLoaded = false;
                        const maxWaitTime = 30000; // 30秒超时
                        const startTime = Date.now();
                        
                        while (!contentLoaded && (Date.now() - startTime) < maxWaitTime) {
                            // 检查多种可能的内容容器
                            const contentSelectors = [
                                '.profiles__timeline__bd',
                                '.search__main',
                                '.timeline',
                                '[data-v-]',
                                '.feed-item',
                                '.status-item',
                                '[class*="timeline"]',
                                '[class*="feed"]',
                                '[class*="post"]'
                            ];
                            
                            let foundContent = false;
                            let contentCount = 0;
                            
                            for (const selector of contentSelectors) {
                                const container = document.querySelector(selector);
                                if (container) {
                                    const textContent = container.textContent || '';
                                    const meaningfulText = textContent.replace(/\\s+/g, ' ').trim();
                                    
                                    // 检查是否有有意义的内容
                                    if (meaningfulText.length > 200 && 
                                        !meaningfulText.includes('默认排序最新讨论') &&
                                        !meaningfulText.includes('loading') &&
                                        (meaningfulText.includes('$') || meaningfulText.includes('股票') || meaningfulText.includes('投资'))) {
                                        foundContent = true;
                                        contentCount++;
                                    }
                                    
                                    // 检查具体的帖子元素
                                    const postElements = container.querySelectorAll('div[class*="item"], div[class*="post"], div[class*="feed"], div[class*="status"]');
                                    if (postElements.length > 3) {
                                        foundContent = true;
                                        contentCount += postElements.length;
                                    }
                                }
                            }
                            
                            if (foundContent && contentCount > 5) {
                                contentLoaded = true;
                                console.log(`✅ 检测到有效内容，元素数量: ${contentCount}`);
                                break;
                            }
                            
                            // 尝试触发内容加载
                            window.scrollTo(0, Math.min(document.body.scrollHeight / 3, 1000));
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }
                        
                        if (!contentLoaded) {
                            console.log('⚠️ 内容加载超时，继续执行...');
                        }
                        """,
                        
                        # 第四步：智能滚动加载更多内容
                        """
                        console.log('📜 开始智能滚动加载更多内容...');
                        
                        let scrollAttempts = 0;
                        const maxScrollAttempts = 10;
                        let lastHeight = document.body.scrollHeight;
                        
                        while (scrollAttempts < maxScrollAttempts) {
                            // 平滑滚动到底部
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            });
                            
                            // 等待新内容加载
                            await new Promise(resolve => setTimeout(resolve, 3000));
                            
                            // 检查页面高度是否增加（表示有新内容加载）
                            const currentHeight = document.body.scrollHeight;
                            if (currentHeight > lastHeight) {
                                console.log(`📈 检测到新内容加载，页面高度: ${lastHeight} -> ${currentHeight}`);
                                lastHeight = currentHeight;
                            } else {
                                // 尝试查找并点击"加载更多"按钮
                                const loadMoreButtons = document.querySelectorAll(
                                    'button, a, div[class*="load"], div[class*="more"], [onclick*="load"], [class*="next"]'
                                );
                                
                                let clickedButton = false;
                                for (const btn of loadMoreButtons) {
                                    const text = (btn.textContent || btn.innerText || '').toLowerCase();
                                    if (text.includes('更多') || text.includes('加载') || 
                                        text.includes('load') || text.includes('more') || 
                                        text.includes('next') || text.includes('继续')) {
                                        try {
                                            btn.click();
                                            console.log('🔄 点击加载更多按钮:', text.substring(0, 20));
                                            clickedButton = true;
                                            await new Promise(resolve => setTimeout(resolve, 2000));
                                            break;
                                        } catch(e) {
                                            console.log('❌ 点击按钮失败:', e.message);
                                        }
                                    }
                                }
                                
                                if (!clickedButton) {
                                    console.log('🛑 未找到更多内容或加载按钮，停止滚动');
                                    break;
                                }
                            }
                            
                            scrollAttempts++;
                        }
                        
                        // 回到顶部
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        
                        console.log('✅ 智能滚动完成');
                        """,
                        
                        # 第五步：最终页面分析和优化
                        """
                        console.log('🔍 执行最终页面分析...');
                        
                        // 页面统计信息
                        const stats = {
                            title: document.title,
                            url: window.location.href,
                            htmlLength: document.documentElement.outerHTML.length,
                            bodyTextLength: document.body.innerText.length
                        };
                        
                        // 分析主要内容区域
                        const mainContainers = {
                            searchMain: !!document.querySelector('.search__main'),
                            timelineBd: !!document.querySelector('.profiles__timeline__bd'),
                            timelineHd: !!document.querySelector('.profiles__timeline__hd')
                        };
                        
                        // 统计可能的帖子元素
                        const postElements = document.querySelectorAll(
                            '.profiles__timeline__bd > *, [class*="status"], [class*="post"], [class*="item"], [class*="card"], [class*="feed"]'
                        );
                        
                        console.log('📊 页面分析结果:', {
                            ...stats,
                            ...mainContainers,
                            postElementsCount: postElements.length,
                            hasStockSymbols: document.body.innerText.includes('$'),
                            hasDiscussion: document.body.innerText.includes('讨论')
                        });
                        
                        console.log('🎉 雪球网站爬取准备完成！');
                        """
                    ],
                    
                    # 等待条件
                    wait_for="css:body",
                    
                    # 页面加载超时
                    page_timeout=60000,  # 60秒
                    
                    # 延迟返回HTML，确保所有异步内容加载完成
                    delay_before_return_html=5.0
                )
                
                # 执行爬取
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    print(f"🚀 第 {attempt + 1} 次尝试爬取: {url}")
                    result = await crawler.arun(url=url, config=run_config)
                    
                    if result.success and result.html:
                        print(f"✅ 爬取成功！HTML长度: {len(result.html)}, Markdown长度: {len(result.markdown) if result.markdown else 0}")
                        
                        # 保存调试HTML
                        with open('/Users/lingxiao/PycharmProjects/TradingAgents/spider/debug_html.html', 'w', encoding='utf-8') as f:
                            f.write(result.html)
                        print("💾 调试HTML已保存")
                        
                        return result.html
                    else:
                        error_msg = result.error_message if hasattr(result, 'error_message') else '未知错误'
                        print(f"❌ 爬取失败: {error_msg}")
                        
            except Exception as e:
                print(f"💥 第 {attempt + 1} 次尝试异常: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = random.uniform(5, 10) * (attempt + 1)
                    print(f"⏰ 等待 {wait_time:.1f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                    
        return None
    
    def extract_posts_from_html(self, html: str, query_name: str) -> List[Dict]:
        """从HTML中提取帖子信息"""
        posts = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            print(f"HTML解析成功，页面标题: {soup.title.string if soup.title else '无标题'}")
            
            # 保存调试用的HTML
            with open('/Users/lingxiao/PycharmProjects/TradingAgents/spider/debug_html.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print("调试HTML已保存到 debug_html.html")
            
            # 雪球网站特定的选择器策略（基于实际页面结构）
            selectors = [
                # 雪球讨论页面的主要内容区域
                '.profiles__timeline__bd > *',
                '.profiles__timeline__bd div',
                '.search__main div',
                # 雪球特定的类名
                'div[class*="timeline"]',
                'div[class*="status"]', 
                'div[class*="feed"]',
                'div[class*="item"]',
                'div[class*="card"]',
                'div[class*="post"]',
                'div[class*="content"]',
                'div[class*="message"]',
                'div[class*="comment"]',
                'div[class*="discuss"]',
                # 通用选择器
                'article',
                '.timeline-item',
                '.status-item', 
                '.feed-item',
                '.post-item',
                '.card-item',
                # React/Vue组件可能的类名
                '[class*="Timeline"]',
                '[class*="Status"]',
                '[class*="Feed"]',
                '[class*="Post"]',
                '[class*="Card"]',
                '[class*="Item"]',
                '[class*="Message"]',
                '[class*="Content"]',
                # 数据属性
                '[data-type="status"]',
                '[data-type="post"]',
                '[data-type="timeline"]',
                '[data-type="item"]',
                # 通用容器
                'section',
                '.container > div',
                '.main > div',
                # 可能包含用户生成内容的元素
                '[class*="user"]',
                '[class*="author"]',
                '[class*="discussion"]'
            ]
            
            all_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"使用选择器 '{selector}' 找到 {len(elements)} 个元素")
                    all_elements.extend(elements)
            
            # 去重
            unique_elements = []
            seen_elements = set()
            for element in all_elements:
                element_id = id(element)
                if element_id not in seen_elements:
                    seen_elements.add(element_id)
                    unique_elements.append(element)
            
            print(f"去重后有 {len(unique_elements)} 个唯一元素")
            
            # 如果仍然没找到，使用更宽泛的搜索
            if not unique_elements:
                print("尝试查找包含关键词的元素...")
                
                # 首先尝试查找雪球特定的结构
                timeline_bd = soup.select_one('.profiles__timeline__bd')
                if timeline_bd:
                    print("找到雪球讨论内容区域")
                    # 获取该区域下的所有直接子元素
                    direct_children = timeline_bd.find_all(recursive=False)
                    print(f"讨论区域直接子元素数量: {len(direct_children)}")
                    unique_elements.extend(direct_children)
                    
                    # 如果直接子元素不够，获取所有子元素
                    if len(unique_elements) < 5:
                        all_children = timeline_bd.find_all('div')
                        print(f"讨论区域所有div子元素数量: {len(all_children)}")
                        unique_elements.extend(all_children)
                
                # 如果还是没找到，使用关键词搜索
                if len(unique_elements) < 5:
                    print("使用关键词搜索...")
                    all_divs = soup.find_all('div')
                    print(f"总共有 {len(all_divs)} 个div元素")
                    
                    for div in all_divs:
                        text = div.get_text(strip=True)
                        # 更宽泛的关键词匹配
                        keywords = [query_name, '股票', '投资', '$', '讨论', '分析', '看多', '看空', '买入', '卖出', '持有']
                        if any(keyword in text for keyword in keywords) and len(text) > 15:
                            unique_elements.append(div)
                            if len(unique_elements) >= 100:  # 增加搜索数量
                                break
            
            print(f"总共找到 {len(unique_elements)} 个可能的帖子项目")
            
            # 按元素文本长度排序，优先处理内容较多的元素
            unique_elements.sort(key=lambda x: len(x.get_text(strip=True)), reverse=True)
            
            # 提取帖子信息
            processed_count = 0
            for i, element in enumerate(unique_elements[:100]):  # 增加处理数量
                try:
                    post_data = self.extract_single_post(element, query_name)
                    if post_data:
                        posts.append(post_data)
                        processed_count += 1
                        print(f"成功提取第 {processed_count} 条帖子 (元素 {i+1}/{len(unique_elements[:100])})")
                        
                        # 输出帖子预览
                        content_preview = post_data['content'][:50] + '...' if len(post_data['content']) > 50 else post_data['content']
                        print(f"  内容预览: {content_preview}")
                        print(f"  用户: {post_data['username']}, 时间: {post_data['time']}")
                        if post_data['stock_tags']:
                            print(f"  股票标签: {post_data['stock_tags']}")
                except Exception as e:
                    if i < 10:  # 只在前10个元素失败时输出详细错误
                        print(f"提取第 {i+1} 个元素失败: {e}")
                    continue
            
            print(f"最终提取到 {len(posts)} 条帖子")
            
            # 如果仍然没有找到帖子，输出更多调试信息
            if not posts:
                print("\n=== 调试信息 ===")
                print(f"页面总字符数: {len(html)}")
                print(f"页面包含查询词 '{query_name}' 的次数: {html.count(query_name)}")
                
                # 查找可能的数据容器
                data_containers = soup.find_all(['script', 'div'], attrs={'id': True})
                print(f"找到 {len(data_containers)} 个可能的数据容器")
                
                for container in data_containers[:5]:
                    if container.name == 'script' and 'application/json' in str(container.get('type', '')):
                        print(f"发现JSON数据脚本: {container.get('id')}")
                    elif container.name == 'div' and any(keyword in container.get('id', '') for keyword in ['app', 'root', 'main']):
                        print(f"发现主要容器: {container.get('id')}, 子元素数: {len(container.find_all())}")
            
        except Exception as e:
            print(f"HTML解析失败: {e}")
            import traceback
            traceback.print_exc()
            
        return posts
    
    def extract_json_data(self, html: str, query_name: str) -> List[Dict]:
        """尝试从页面的JSON数据中提取帖子信息"""
        posts = []
        
        try:
            # 查找可能包含数据的script标签
            soup = BeautifulSoup(html, 'html.parser')
            script_tags = soup.find_all('script')
            
            for script in script_tags:
                script_content = script.string or ''
                
                # 查找可能的JSON数据模式
                json_patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.+?});',
                    r'window\.__NUXT__\s*=\s*({.+?});',
                    r'window\.__DATA__\s*=\s*({.+?});',
                    r'window\.g_initialProps\s*=\s*({.+?});',
                    r'__INITIAL_DATA__\s*=\s*({.+?});',
                    r'window\.SNB\s*=\s*({.+?});'
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, script_content, re.DOTALL)
                    for match in matches:
                        try:
                            data = json.loads(match)
                            extracted_posts = self.parse_json_for_posts(data, query_name)
                            posts.extend(extracted_posts)
                            print(f"从JSON数据中提取到 {len(extracted_posts)} 条帖子")
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"解析JSON数据失败: {e}")
                            continue
            
        except Exception as e:
            print(f"提取JSON数据失败: {e}")
        
        return posts
    
    def parse_json_for_posts(self, data: dict, query_name: str) -> List[Dict]:
        """递归解析JSON数据查找帖子信息"""
        posts = []
        
        def recursive_search(obj, path=""):
            if isinstance(obj, dict):
                # 查找可能的帖子数组
                for key, value in obj.items():
                    if key in ['list', 'data', 'items', 'posts', 'statuses', 'timeline', 'feeds']:
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    post = self.extract_post_from_json_item(item, query_name)
                                    if post:
                                        posts.append(post)
                    recursive_search(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    recursive_search(item, f"{path}[{i}]")
        
        recursive_search(data)
        return posts
    
    def extract_post_from_json_item(self, item: dict, query_name: str) -> Optional[Dict]:
        """从JSON项目中提取帖子信息"""
        try:
            # 查找文本内容
            text_fields = ['text', 'content', 'description', 'title', 'body', 'message']
            content = ""
            for field in text_fields:
                if field in item and isinstance(item[field], str):
                    content = item[field]
                    break
            
            if not content or len(content) < 10:
                return None
            
            # 检查是否包含查询关键词
            keywords = [query_name, '$', '股票', '投资', '讨论', '分析']
            if not any(keyword in content for keyword in keywords):
                return None
            
            # 提取用户信息
            username = "未知用户"
            user_fields = ['user', 'author', 'username', 'screen_name', 'name']
            for field in user_fields:
                if field in item:
                    if isinstance(item[field], dict):
                        username = item[field].get('name', item[field].get('screen_name', '未知用户'))
                    elif isinstance(item[field], str):
                        username = item[field]
                    break
            
            # 提取时间信息
            post_time = "未知时间"
            time_fields = ['created_at', 'time', 'timestamp', 'date', 'publish_time']
            for field in time_fields:
                if field in item and item[field]:
                    post_time = str(item[field])
                    break
            
            # 提取股票标签
            stock_tags = re.findall(r'\$([^$\s]{1,10})\$', content)
            
            # 提取帖子URL
            post_url = None
            url_fields = ['url', 'link', 'target_url', 'href', 'target', 'path']
            for field in url_fields:
                if field in item and isinstance(item[field], str) and item[field]:
                    url = item[field]
                    # 转换为绝对URL
                    if url.startswith('/'):
                        url = f"https://xueqiu.com{url}"
                    elif not url.startswith('http'):
                        url = f"https://xueqiu.com/{url}"
                    
                    # 验证是否是有效的帖子链接
                    if self.is_valid_post_url(url):
                        post_url = url
                        break
            
            # 如果没有找到URL，尝试从ID构建
            if not post_url:
                id_fields = ['id', 'status_id', 'statusId', 'post_id', 'postId']
                for field in id_fields:
                    if field in item and str(item[field]).isdigit():
                        post_id = item[field]
                        # 尝试构建状态URL
                        post_url = f"https://xueqiu.com/statuses/{post_id}"
                        break
            
            # 构建帖子数据
            post_data = {
                'username': username,
                'content': content[:300],
                'time': self.format_time_to_standard(post_time),
                'stock_tags': stock_tags,
                'url': post_url if post_url else f"https://xueqiu.com/k?q={quote(query_name)}#/timeline",
                'is_demo': False,
                'source': 'json_extraction'
            }
            
            return post_data
            
        except Exception as e:
            print(f"从JSON项目提取帖子失败: {e}")
            return post_data
    
    def extract_post_url(self, element) -> Optional[str]:
        """从元素中提取帖子详细页面的URL"""
        try:
            # 雪球网站的帖子链接模式
            link_selectors = [
                'a[href*="/statuses/"]',  # 雪球状态链接
                'a[href*="/status/"]',    # 雪球状态链接变体
                'a[href*="/u/"]',         # 用户页面链接
                'a[href*="/user/"]',      # 用户页面链接变体
                'a[href^="/"][href*="/"]', # 相对链接
                'a[href*="xueqiu.com"]',   # 绝对链接
                'a[title]',                # 带标题的链接
                'a[data-url]',             # 数据URL属性
                'a'                        # 所有链接作为最后备选
            ]
            
            for selector in link_selectors:
                links = element.select(selector)
                for link in links:
                    href = link.get('href', '')
                    data_url = link.get('data-url', '')
                    
                    # 优先使用href
                    url = href or data_url
                    if not url:
                        continue
                    
                    # 过滤掉不相关的链接
                    skip_patterns = [
                        '/search', '/k?q=', '/login', '/register', '/about',
                        '/help', '/privacy', '/terms', '/contact', '/api',
                        'javascript:', 'mailto:', 'tel:', '#'
                    ]
                    
                    if any(pattern in url for pattern in skip_patterns):
                        continue
                    
                    # 转换为绝对URL
                    if url.startswith('/'):
                        url = f"https://xueqiu.com{url}"
                    elif not url.startswith('http'):
                        url = f"https://xueqiu.com/{url}"
                    
                    # 验证是否是有效的帖子链接
                    if self.is_valid_post_url(url):
                        return url
            
            return None
            
        except Exception as e:
            print(f"提取帖子URL失败: {e}")
            return None
    
    def is_valid_post_url(self, url: str) -> bool:
        """验证是否是有效的帖子URL"""
        try:
            # 雪球帖子URL的常见模式
            valid_patterns = [
                r'/statuses/\d+',      # 状态ID
                r'/status/\d+',       # 状态ID变体
                r'/\d+/\d+',          # 用户ID/帖子ID
                r'/u/\d+',            # 用户页面
                r'/user/\d+',         # 用户页面变体
            ]
            
            for pattern in valid_patterns:
                if re.search(pattern, url):
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def crawl_post_detail(self, post_url: str) -> Optional[Dict]:
        """爬取帖子详细页面的完整内容和准确时间"""
        try:
            print(f"正在爬取帖子详细页面: {post_url}")
            
            # 使用相同的爬取配置
            html = await self.crawl_with_retry(post_url)
            if not html:
                return None
            
            # 解析详细页面内容
            soup = BeautifulSoup(html, 'html.parser')
            
            # 提取准确的发布时间
            post_time = "未知时间"
            
            # 优先从.time元素提取时间（雪球帖子详细页面特有）
            time_elem = soup.select_one('.time')
            if time_elem:
                # 检查data-created_at属性
                timestamp_str = time_elem.get('data-created_at')
                if timestamp_str and timestamp_str.isdigit():
                    try:
                        timestamp = int(timestamp_str)
                        # 处理毫秒时间戳
                        if timestamp > 10**12:
                            timestamp = timestamp // 1000
                        dt = datetime.fromtimestamp(timestamp)
                        post_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"✅ 从详细页面data-created_at提取时间: {post_time}")
                    except (ValueError, OSError) as e:
                        print(f"⚠️ 时间戳转换失败: {e}")
                
                # 如果没有时间戳，尝试datetime属性
                if post_time == "未知时间":
                    datetime_attr = time_elem.get('datetime')
                    if datetime_attr:
                        try:
                            if 'T' in datetime_attr:
                                dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                                post_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                print(f"✅ 从详细页面datetime属性提取时间: {post_time}")
                        except ValueError:
                            pass
                
                # 最后尝试文本内容
                if post_time == "未知时间":
                    time_text = time_elem.get_text(strip=True)
                    if time_text and '发布于' in time_text:
                        # 提取"发布于2025-07-02 21:40"中的时间
                        time_match = re.search(r'发布于(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', time_text)
                        if time_match:
                            post_time = time_match.group(1) + ':00'  # 添加秒数
                            print(f"✅ 从详细页面文本提取时间: {post_time}")
            
            # 如果还没找到时间，使用其他选择器
            if post_time == "未知时间":
                time_selectors = [
                    '[data-created_at]', 'time', '[datetime]', '.timestamp',
                    '.post-time', '.created-at', '.publish-time', '[title*=":"]'
                ]
                
                for selector in time_selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        timestamp_str = elem.get('data-created_at')
                        if timestamp_str and timestamp_str.isdigit():
                            try:
                                timestamp = int(timestamp_str)
                                if timestamp > 10**12:
                                    timestamp = timestamp // 1000
                                dt = datetime.fromtimestamp(timestamp)
                                post_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                print(f"✅ 从{selector}提取时间: {post_time}")
                                break
                            except (ValueError, OSError):
                                pass
            
            # 雪球帖子详细页面的内容选择器
            content_selectors = [
                '.status-content',           # 状态内容
                '.detail-content',           # 详细内容
                '.post-content',             # 帖子内容
                '.article-content',          # 文章内容
                '[class*="content"]',        # 包含content的类
                '[class*="text"]',           # 包含text的类
                '[class*="desc"]',           # 包含desc的类
                '.timeline-item-content',    # 时间线项目内容
                'article',                   # 文章标签
                'main',                      # 主要内容区域
                '.main-content',             # 主要内容
                'p',                         # 段落
                'div[data-content]'          # 数据内容属性
            ]
            
            full_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > len(full_content):
                        full_content = content_text
            
            # 如果没有找到特定内容，尝试从整个页面提取
            if not full_content or len(full_content) < 50:
                # 移除脚本和样式标签
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 获取页面主体内容
                body = soup.find('body')
                if body:
                    full_content = body.get_text(strip=True)
            
            # 清理内容
            if full_content:
                # 移除多余的空白字符
                full_content = re.sub(r'\s+', ' ', full_content).strip()
                
                # 移除一些无用的文本模式
                remove_patterns = [
                    r'点击查看更多.*?(?=\s|$)',
                    r'展开全文.*?(?=\s|$)',
                    r'收起.*?(?=\s|$)',
                    r'\d+赞\s*\d+评论.*?(?=\s|$)',
                    r'分享.*?(?=\s|$)',
                    r'举报.*?(?=\s|$)',
                    r'登录.*?(?=\s|$)',
                    r'注册.*?(?=\s|$)'
                ]
                
                for pattern in remove_patterns:
                    full_content = re.sub(pattern, '', full_content)
                
                full_content = full_content.strip()
                
                # 使用优化后的无关内容过滤器
                if self.is_irrelevant_content(full_content):
                    print(f"  检测到无关内容，跳过")
                    return None
                print(f"✅ 成功提取内容，长度: {len(full_content)}")
            
            # 返回内容和时间
            result = {
                'content': full_content if full_content and len(full_content) > 20 else None,
                'time': post_time
            }
            
            return result
            
        except Exception as e:
            print(f"爬取帖子详细内容失败: {e}")
            return None
    
    def extract_single_post(self, element, query_name: str) -> Optional[Dict]:
        """提取单个帖子的信息，包括详细页面链接"""
        try:
            text = element.get_text(strip=True)
            
            # 过滤太短的内容或明显的导航元素
            if len(text) < 10 or text in ['默认排序', '最新讨论', '讨论最多', '综合', '讨论', '股票', '组合', '用户']:
                return None
            
            # 检查是否包含查询关键词或相关内容
            keywords = [query_name, '$', '股票', '投资', '讨论', '分析', '看多', '看空', '买入', '卖出', '持有', '涨', '跌', '行情', '市场', '财报', '业绩']
            if not any(keyword in text for keyword in keywords):
                return None
            
            # 提取帖子详细页面链接
            post_url = self.extract_post_url(element)
            
            # 提取用户名 - 雪球网站特定的选择器
            username = "未知用户"
            username_selectors = [
                # 雪球特定的用户链接模式
                'a[href*="/u/"]', 'a[href*="/user/"]', 'a[href*="/profile/"]',
                'a[href^="/"][href*="/"]',  # 雪球用户链接通常是 /数字ID 格式
                # 通用用户名选择器
                '[class*="user"]', '[class*="author"]', '[class*="name"]', 
                '.username', '.author', '.user-name', '.author-name',
                '[class*="User"]', '[class*="Author"]', '[class*="Name"]',
                '[data-user]', '[data-author]', 'span[class*="user"]',
                'div[class*="user"]'
            ]
            
            for selector in username_selectors:
                user_elem = element.select_one(selector)
                if user_elem:
                    username_text = user_elem.get_text(strip=True)
                    # 验证用户名的合理性
                    if (username_text and 
                        len(username_text) < 50 and 
                        username_text not in ['更多', '加载', '展开', '收起', '分享', '评论', '点赞'] and
                        not username_text.isdigit()):
                        username = username_text
                        break
            
            # 如果没找到用户名，尝试正则提取
            if username == "未知用户":
                user_patterns = [
                    r'@([\w\u4e00-\u9fa5]{2,20})',  # @用户名
                    r'作者[：:](\S+)',  # 作者：用户名
                    r'发布者[：:](\S+)',  # 发布者：用户名
                    r'用户[：:](\S+)'   # 用户：用户名
                ]
                for pattern in user_patterns:
                    user_match = re.search(pattern, text)
                    if user_match:
                        username = user_match.group(1)
                        break
            
            # 提取时间 - 优先使用雪球网站的时间戳
            post_time = "未知时间"
            
            # 首先尝试从data-created_at属性获取时间戳（雪球网站特有）
            time_elem_with_data = element.select_one('[data-created_at]')
            if time_elem_with_data:
                timestamp_str = time_elem_with_data.get('data-created_at')
                if timestamp_str and timestamp_str.isdigit():
                    try:
                        timestamp = int(timestamp_str)
                        # 处理毫秒时间戳
                        if timestamp > 10**12:
                            timestamp = timestamp // 1000
                        dt = datetime.fromtimestamp(timestamp)
                        post_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"✅ 从data-created_at提取时间: {post_time}")
                    except (ValueError, OSError) as e:
                        print(f"⚠️ 时间戳转换失败: {e}")
            
            # 如果没有找到时间戳，使用传统的时间选择器
            if post_time == "未知时间":
                time_selectors = [
                    # 雪球特定的时间选择器
                    '[class*="time"]', '[class*="date"]', 'time', '[title*=":"]',
                    '.timestamp', '.post-time', '.created-at', '.publish-time',
                    '[class*="Time"]', '[class*="Date"]', '[class*="Timestamp"]',
                    '[data-time]', '[data-date]', 'span[class*="time"]',
                    'div[class*="time"]', '.time-ago', '.relative-time',
                    # 可能包含时间的span或small元素
                    'span[title]', 'small[title]', '.meta', '.info'
                ]
                
                for selector in time_selectors:
                    time_elem = element.select_one(selector)
                    if time_elem:
                        # 优先检查datetime属性
                        datetime_attr = time_elem.get('datetime')
                        if datetime_attr:
                            try:
                                # 解析ISO格式时间
                                if 'T' in datetime_attr:
                                    dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                                    # 转换为本地时间
                                    post_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    print(f"✅ 从datetime属性提取时间: {post_time}")
                                    break
                            except ValueError:
                                pass
                        
                        # 然后检查title属性
                        title_attr = time_elem.get('title')
                        if title_attr and len(title_attr) < 50:
                            post_time = title_attr
                            print(f"✅ 从title属性提取时间: {post_time}")
                            break
                        
                        # 最后检查文本内容
                        time_text = time_elem.get_text(strip=True)
                        if time_text and len(time_text) < 50:
                            post_time = time_text
                            print(f"✅ 从文本内容提取时间: {post_time}")
                            break
            
            # 如果没找到时间，尝试正则提取
            if post_time == "未知时间":
                time_patterns = [
                    r'(\d+分钟前)',
                    r'(\d+小时前)', 
                    r'(\d+天前)',
                    r'(\d+周前)',
                    r'(\d+个月前)',
                    r'(昨天|前天)',
                    r'(\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})',
                    r'(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})',
                    r'(\d{1,2}月\d{1,2}日\s+\d{1,2}:\d{2})',
                    r'(\d{4}年\d{1,2}月\d{1,2}日)',
                    r'(今天\s+\d{1,2}:\d{2})',
                    r'(刚刚|刚才)'
                ]
                for pattern in time_patterns:
                    time_match = re.search(pattern, text)
                    if time_match:
                        post_time = time_match.group(1)
                        break
            
            # 提取股票标签 - 使用更全面的模式
            stock_tags = []
            stock_patterns = [
                r'\$([^$\s]{1,10})\$',  # $股票代码$
                r'\$([A-Z]{1,5})\b',    # $大写字母代码
                r'([A-Z]{2,5})\.',      # 股票代码.
                r'#([^#\s]{1,10})#'     # #股票标签#
            ]
            
            for pattern in stock_patterns:
                matches = re.findall(pattern, text)
                stock_tags.extend(matches)
            
            # 去重股票标签
            stock_tags = list(set(stock_tags))
            
            # 提取主要内容 - 尝试找到最相关的文本
            content_selectors = [
                '[class*="content"]', '[class*="text"]', '[class*="desc"]',
                '.post-content', '.message-content', '.status-content',
                '[class*="Content"]', '[class*="Text"]', '[class*="Message"]',
                'p', '.description', '.summary'
            ]
            
            main_content = text
            for selector in content_selectors:
                content_elem = element.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 20:  # 确保内容有意义
                        main_content = content_text
                        break
            
            # 清理内容
            content = main_content[:300]  # 增加长度限制
            content = re.sub(r'\s+', ' ', content).strip()
            
            # 移除一些无用的文本
            remove_patterns = [
                r'点击查看更多.*',
                r'展开全文.*',
                r'收起.*',
                r'\d+赞\s*\d+评论.*',
                r'分享.*',
                r'举报.*'
            ]
            
            for pattern in remove_patterns:
                content = re.sub(pattern, '', content)
            
            content = content.strip()
            
            # 确保内容有意义
            if len(content) < 5:
                return None
            
            # 过滤掉页面无关内容
            if self.is_irrelevant_content(content):
                return None
            
            # 构建帖子数据
            post_data = {
                'username': username,
                'content': content,
                'time': self.format_time_to_standard(post_time),
                'stock_tags': stock_tags,
                'url': post_url if post_url else f"https://xueqiu.com/k?q={quote(query_name)}#/timeline",
                'is_demo': False,
                'element_class': element.get('class', []) if hasattr(element, 'get') else [],
                'element_id': element.get('id', '') if hasattr(element, 'get') else ''
            }
            
            return post_data
            
        except Exception as e:
            print(f"提取帖子信息失败: {e}")
            return None
    
    async def crawl_xueqiu_data(self, query_name: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """爬取雪球数据的主函数"""
        print(f"开始爬取雪球网站关于 '{query_name}' 的讨论数据...")
        
        url = self.format_url(query_name)
        print(f"开始爬取: {url}")
        print(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 使用crawl_with_retry方法，该方法已包含爬虫配置
        
        # 爬取HTML
        html = await self.crawl_with_retry(url)
        if not html:
            print("无法获取页面内容")
            return []
        
        print(f"获取到HTML内容，长度: {len(html)} 字符")
        
        # 保存调试HTML
        with open('/Users/lingxiao/PycharmProjects/TradingAgents/spider/debug_html.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print("调试HTML已保存到 debug_html.html")
        
        # 多种方式提取帖子
        all_posts = []
        
        # 1. 从HTML DOM结构提取
        print("\n=== 尝试从HTML DOM结构提取帖子 ===")
        html_posts = self.extract_posts_from_html(html, query_name)
        all_posts.extend(html_posts)
        print(f"从HTML DOM提取到 {len(html_posts)} 条帖子")
        
        # 2. 从JSON数据提取
        print("\n=== 尝试从JSON数据提取帖子 ===")
        json_posts = self.extract_json_data(html, query_name)
        all_posts.extend(json_posts)
        print(f"从JSON数据提取到 {len(json_posts)} 条帖子")
        
        # 如果没有提取到任何帖子，尝试更宽泛的搜索
        if not all_posts:
            print("\n=== 未找到帖子，尝试更宽泛的搜索 ===")
            soup = BeautifulSoup(html, 'html.parser')
            
            # 查找所有可能包含文本的元素
            text_elements = soup.find_all(['p', 'div', 'span', 'article', 'section'])
            print(f"找到 {len(text_elements)} 个可能包含文本的元素")
            
            for element in text_elements:
                text = element.get_text(strip=True)
                if len(text) > 30 and any(keyword in text for keyword in [query_name, '股票', '投资', '$']):
                    post_data = {
                        'username': "未知用户",
                        'content': text[:300],  # 限制长度
                        'time': "未知时间",
                        'stock_tags': re.findall(r'\$([^$\s]{1,10})\$', text),
                        'url': url,
                        'is_demo': False,
                        'element_type': element.name
                    }
                    all_posts.append(post_data)
                    if len(all_posts) >= 20:  # 限制数量
                        break
            
            print(f"宽泛搜索找到 {len(all_posts)} 条可能的帖子")
        
        # 去重（基于内容相似度）
        unique_posts = self.deduplicate_posts(all_posts)
        print(f"去重后共有 {len(unique_posts)} 条唯一帖子")
        
        # 深度爬取：获取每个帖子的完整内容
        print("\n=== 开始深度爬取帖子详细内容 ===")
        enhanced_posts = []
        # 处理所有帖子以获取准确的时间信息
        limited_posts = unique_posts[:10]  # 处理前10条帖子
        print(f"处理前 {len(limited_posts)} 条帖子以获取准确时间信息")
        
        for i, post in enumerate(limited_posts):
            try:
                print(f"正在处理第 {i+1}/{len(limited_posts)} 条帖子...")
                
                # 如果有详细页面链接，爬取完整内容和准确时间
                # 雪球URL格式：https://xueqiu.com/用户ID/帖子ID
                post_url = post.get('url', '')
                print(f"  检查URL: {post_url}")
                
                url_has_xueqiu = 'xueqiu.com' in post_url
                url_has_statuses = '/statuses/' in post_url
                url_matches_pattern = bool(re.match(r'https://xueqiu\.com/\d+/\d+', post_url))
                
                print(f"  URL检查结果: xueqiu={url_has_xueqiu}, statuses={url_has_statuses}, pattern={url_matches_pattern}")
                
                if (post_url and url_has_xueqiu and (url_has_statuses or url_matches_pattern)):
                    print(f"  ✅ URL匹配成功，开始爬取详细页面: {post_url}")
                    detail_result = await self.crawl_post_detail(post_url)
                    if detail_result and detail_result.get('content'):
                        full_content = detail_result['content']
                        detail_time = detail_result['time']
                        
                        if len(full_content) > len(post['content']):
                            print(f"  成功获取完整内容，长度从 {len(post['content'])} 增加到 {len(full_content)}")
                            post['content'] = full_content
                        
                        # 更新时间信息（如果从详细页面获取到了准确时间）
                        if detail_time != "未知时间":
                            print(f"  更新时间信息：从 {post['time']} 更新为 {detail_time}")
                            post['time'] = detail_time
                        
                        post['content_enhanced'] = True
                    else:
                        print(f"  未能获取到更完整的内容，保持原内容")
                        post['content_enhanced'] = False
                else:
                    print(f"  ❌ URL不匹配，跳过详细页面爬取")
                    post['content_enhanced'] = False
                
                enhanced_posts.append(post)
                
                # 添加延迟避免请求过快
                if i < len(limited_posts) - 1:  # 最后一个不需要延迟
                    await asyncio.sleep(random.uniform(1, 2))  # 减少延迟时间
                    
            except Exception as e:
                print(f"  处理第 {i+1} 条帖子时出错: {e}")
                enhanced_posts.append(post)  # 即使出错也保留原始帖子
                continue
        
        # 将未处理的帖子也加入结果（不进行深度爬取）
        for post in unique_posts[10:]:
            post['content_enhanced'] = False
            enhanced_posts.append(post)
        
        print(f"深度爬取完成，共处理 {len(enhanced_posts)} 条帖子，其中 {len(limited_posts)} 条进行了深度爬取")
        
        # 过滤时间范围
        filtered_posts = []
        for post in enhanced_posts:
            post_datetime = self.parse_time_string(post['time'])
            if post_datetime and self.is_in_time_range(post_datetime, start_time, end_time):
                filtered_posts.append(post)
            elif post['time'] == "未知时间":  # 如果时间未知，也包含进来
                filtered_posts.append(post)
        
        print(f"\n=== 最终结果 ===")
        print(f"共爬取到 {len(unique_posts)} 条帖子，时间范围内有 {len(filtered_posts)} 条")
        
        # 如果结果为空，添加一个调试信息帖子
        if not filtered_posts:
            debug_post = {
                'username': "调试信息",
                'content': f"未能从雪球网站提取到关于 '{query_name}' 的帖子。请检查网站结构或调整爬虫参数。",
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'stock_tags': [],
                'url': url,
                'is_demo': False
            }
            filtered_posts.append(debug_post)
        
        return filtered_posts
    
    def deduplicate_posts(self, posts: List[Dict]) -> List[Dict]:
        """去除重复的帖子"""
        unique_posts = []
        seen_contents = set()
        
        for post in posts:
            # 使用内容的前100个字符作为去重标准
            content_key = post['content'][:100].strip()
            if content_key not in seen_contents and len(content_key) > 10:
                seen_contents.add(content_key)
                unique_posts.append(post)
        
        return unique_posts
    
    def save_results(self, posts: List[Dict], filename: str = None) -> str:
        """保存爬取结果"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'/Users/lingxiao/PycharmProjects/TradingAgents/spider/xueqiu_data_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {filename}")
        return filename

# 示例代理列表（需要替换为真实可用的代理）
SAMPLE_PROXIES = [
    # 'http://proxy1.example.com:8080',
    # 'http://proxy2.example.com:8080',
    # 'socks5://proxy3.example.com:1080'
]

async def main():
    """主函数示例"""
    # 创建爬虫实例
    crawler = XueqiuEnhancedCrawler(
        use_proxy=False,  # 设置为True启用代理
        proxy_list=SAMPLE_PROXIES
    )
    
    # 设置爬取参数（缩短时间范围以加快测试）
    query_name = "特斯拉"  # 查询名称
    start_time = datetime.now() - timedelta(days=1)  # 开始时间（1天前，缩短测试时间）
    end_time = datetime.now()  # 结束时间（现在）
    
    try:
        # 执行爬取
        posts = await crawler.crawl_xueqiu_data(query_name, start_time, end_time)
        
        if posts:
            # 保存结果
            filename = crawler.save_results(posts)
            
            # 输出结果
            print("\n<results>")
            for i, post in enumerate(posts, 1):
                print(f"\n帖子 {i}:")
                print(f"用户: {post['username']}")
                print(f"时间: {post['time']}")
                print(f"内容: {post['content']}")
                if post['stock_tags']:
                    print(f"股票标签: {', '.join(['$' + tag + '$' for tag in post['stock_tags']])})")
                print(f"链接: {post['url']}")
            print("\n</results>")
            
            # 统计信息
            print(f"\n统计信息:")
            print(f"- 总帖子数: {len(posts)}")
            print(f"- 包含股票标签的帖子: {sum(1 for p in posts if p['stock_tags'])}")
            print(f"- 平均内容长度: {sum(len(p['content']) for p in posts) // len(posts) if posts else 0}字符")
            
        else:
            print("未爬取到任何数据")
            
    except Exception as e:
        print(f"爬取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())