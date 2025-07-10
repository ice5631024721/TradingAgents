import tushare as ts
# 获取实时新闻（默认80条）
news = ts.get_latest_news()
# 筛选港股/美股新闻（classify字段包含"港股"或"美股"）
hk_news = news[news['classify'].str.contains('港股')]
us_news = news[news['classify'].str.contains('美股')]
# 获取详细内容（需设置show_content=True）
detailed_news = ts.get_latest_news(top=10, show_content=True)