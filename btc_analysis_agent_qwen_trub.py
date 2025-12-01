import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
# 移除talib依赖，使用pandas自己实现技术指标

warnings.filterwarnings('ignore')  # 忽略ARIMA模型的一些警告信息

# 新增：从binance导入Client以获取实时价格
from binance import Client

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# 配置数据库连接 - 使用与比特币数据相同的数据库配置
db_config = {
    'host': 'audit.mingyuanyun.com',  # 数据库主机地址
    'port': 63306,                    # 数据库端口
    'user': '385b7dce-81e1-44fe-b6a6-23148bfac73a',       # 数据库用户名
    'password': 'BAauLPFdkzO47RuC',  # 数据库密码
    'database': 'mycommunity_config_test',  # 数据库名称
    'charset': 'utf8mb4'
}

# 初始化Binance客户端，无需API Key即可访问公开数据
client = Client()

# ====== 比特币助手 system prompt 和函数描述 ======
system_prompt = """我是比特币价格分析助手，以下是关于比特币价格数据表的字段信息，我可以编写SQL查询并分析比特币价格数据

-- 比特币价格数据表
here is the btc price table info:
CREATE TABLE btc_usdt_kline (
    日期 DATE,
    开盘时间 DATETIME,
    开盘价 DECIMAL(18,8),
    最高价 DECIMAL(18,8),
    最低价 DECIMAL(18,8),
    收盘价 DECIMAL(18,8),
    成交量 DECIMAL(20,8),
    收盘时间 DATETIME,
    PRIMARY KEY (日期, 开盘时间)
);

我将回答用户关于比特币价格相关的问题，包括价格走势分析、交易量分析、价格波动分析等。
我还可以获取比特币的实时价格数据（精确到秒）和使用ARIMA模型进行价格预测。

每当获取到工具返回的实时价格数据、SQL查询结果或ARIMA预测结果时，我会基于这些数据进行进一步的分析和思考，提供更有价值的洞察和建议。

对于实时价格数据，我会重点关注：
1. 当前价格与历史价格的对比分析
2. 短期价格走势的技术面解读
3. 潜在的投资机会和风险点
4. 基于当前市场状况的策略建议

每当 exc_sql 工具返回 markdown 表格和图片时，我必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
    {
        "name": "arima_stock",
        "description": "使用ARIMA模型对指定币子未来N天的价格进行预测",
        "parameters": {
            "type": "object",
            "properties": {
                "b_code": {
                    "type": "string",
                    "description": "币子代码，必填",
                },
                "n": {
                    "type": "integer",
                    "description": "预测的天数",
                    "default": 7
                }
            },
            "required": ["b_code"],
        },
    },
    {
        "name": "get_real_time_price",
        "description": "获取指定币子的实时价格数据，精确到秒",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "交易对符号，如BTCUSDT，必填",
                }
            },
            "required": ["symbol"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    优化功能：检查数据库历史数据是否有缺失，如有缺失则从交易所获取并更新数据库
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def check_and_update_data(self, engine):
        """
        检查数据库中的数据是否有缺失，如果有缺失则从Binance获取并更新
        """
        try:
            # 查询数据库中最新的数据日期
            latest_date_query = "SELECT MAX(日期) as latest_date FROM btc_usdt_kline"
            latest_date_result = pd.read_sql(latest_date_query, engine)
            latest_date = latest_date_result['latest_date'].iloc[0]
            
            # 获取当前日期
            current_date = datetime.now().date()
            
            # 如果数据库中没有数据或数据有缺失
            if latest_date is None or latest_date < current_date:
                # 计算需要获取数据的起始时间
                if latest_date is None:
                    # 如果没有数据，获取过去30天的数据
                    start_date = current_date - timedelta(days=30)
                else:
                    # 如果有数据缺失，从最新日期的下一天开始获取
                    start_date = latest_date + timedelta(days=1)
                
                print(f"检测到数据缺失，需要从 {start_date} 开始更新数据到 {current_date}")
                
                # 从Binance获取数据
                # 注意：Binance API有限制，单次获取的数据量不能太大
                # 这里我们按天获取数据
                missing_data = []
                current_fetch_date = start_date
                
                while current_fetch_date <= current_date:
                    try:
                        # 计算结束日期（最多获取7天的数据，避免API限制）
                        end_fetch_date = min(current_fetch_date + timedelta(days=6), current_date)
                        
                        # 获取K线数据，使用1天间隔
                        klines = client.get_historical_klines(
                            symbol='BTCUSDT',
                            interval=Client.KLINE_INTERVAL_1DAY,
                            start_str=current_fetch_date.strftime('%Y-%m-%d'),
                            end_str=end_fetch_date.strftime('%Y-%m-%d')
                        )
                        
                        if klines:
                            # 转换为DataFrame
                            df_batch = pd.DataFrame(klines, columns=[
                                '开盘时间戳', '开盘价', '最高价', '最低价', '收盘价', '成交量',
                                '收盘时间戳', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'
                            ])
                            
                            # 处理数据类型和时间格式
                            df_batch['开盘时间'] = pd.to_datetime(df_batch['开盘时间戳'], unit='ms')
                            df_batch['收盘时间'] = pd.to_datetime(df_batch['收盘时间戳'], unit='ms')
                            df_batch['日期'] = df_batch['开盘时间'].dt.date
                            
                            # 转换数值类型
                            numeric_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
                            for col in numeric_columns:
                                df_batch[col] = df_batch[col].astype(float)
                            
                            # 选择需要的列
                            df_batch = df_batch[['日期', '开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '收盘时间']]
                            
                            # 添加到缺失数据列表
                            missing_data.append(df_batch)
                        
                        # 更新下一次获取的起始日期
                        current_fetch_date = end_fetch_date + timedelta(days=1)
                        
                        # 添加短暂延迟，避免触发API限制
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"获取 {current_fetch_date} 到 {end_fetch_date} 的数据时出错: {str(e)}")
                        current_fetch_date = end_fetch_date + timedelta(days=1)
                        continue
                
                # 如果有缺失数据需要写入数据库
                if missing_data:
                    # 合并所有批次的数据
                    full_missing_data = pd.concat(missing_data, ignore_index=True)
                    
                    # 写入数据库
                    with engine.begin() as conn:
                        # 使用 append 模式，避免覆盖已有数据
                        full_missing_data.to_sql(
                            name='btc_usdt_kline',
                            con=conn,
                            if_exists='append',
                            index=False
                        )
                    
                    print(f"成功更新 {len(full_missing_data)} 条数据到数据库")
                    return f"数据更新成功：新增 {len(full_missing_data)} 条记录"
                else:
                    print("没有检测到需要更新的数据")
                    return "数据库数据已经是最新的"
            else:
                print("数据库数据已经是最新的")
                return "数据库数据已经是最新的"
        
        except Exception as e:
            print(f"检查和更新数据时出错: {str(e)}")
            # 即使更新失败，也不阻止后续查询
            return f"数据更新检查失败: {str(e)}，但将继续执行查询"

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        args = json.loads(params)
        sql_input = args['sql_input']
        database = args.get('database', db_config['database'])
        
        # 使用sqlalchemy创建数据库连接
        connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{database}?charset=utf8mb4"
        engine = create_engine(connection_string)
        
        try:
            # 首先检查并更新数据
            update_message = self.check_and_update_data(engine)
            
            # 然后执行用户的SQL查询
            df = pd.read_sql(sql_input, engine)
            md = df.head(10).to_markdown(index=False)
            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'btc_images')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'btc_chart_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 生成图表
            generate_btc_chart(df, save_path)
            img_path = os.path.join('btc_images', filename)
            img_md = f'![比特币图表]({img_path})'
            
            # 返回查询结果，同时包含数据更新的信息
            return f"## 数据更新状态\n{update_message}\n\n## 查询结果\n{md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

# ========== 比特币数据可视化函数 ========== 
def generate_btc_chart(df_sql, save_path):
    columns = df_sql.columns
    
    # 如果有日期或时间列，设置为索引
    date_columns = []
    for col in columns:
        if '日期' in col or '时间' in col:
            date_columns.append(col)
    
    # 如果有价格相关列
    price_columns = []
    for col in columns:
        if any(x in col for x in ['开盘价', '收盘价', '最高价', '最低价', '价格']):
            price_columns.append(col)
    
    # 如果有成交量相关列
    volume_columns = []
    for col in columns:
        if '成交量' in col:
            volume_columns.append(col)
    
    # 创建图表
    fig = plt.figure(figsize=(12, 8))
    
    # 如果有价格列，绘制价格走势图
    if price_columns and date_columns:
        ax1 = fig.add_subplot(211)
        date_col = date_columns[0]
        
        # 绘制价格线
        for price_col in price_columns:
            ax1.plot(df_sql[date_col], df_sql[price_col], label=price_col, linewidth=2)
        
        ax1.set_title('比特币价格走势')
        ax1.set_ylabel('价格 (USDT)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 如果有成交量列，绘制成交量
        if volume_columns:
            ax2 = fig.add_subplot(212)
            for vol_col in volume_columns:
                ax2.bar(df_sql[date_col], df_sql[vol_col], label=vol_col, alpha=0.7, color='orange')
            
            ax2.set_title('比特币成交量')
            ax2.set_xlabel('日期/时间')
            ax2.set_ylabel('成交量')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
    
    # 如果没有价格列但有成交量列
    elif volume_columns and date_columns:
        ax1 = fig.add_subplot(111)
        date_col = date_columns[0]
        
        for vol_col in volume_columns:
            ax1.bar(df_sql[date_col], df_sql[vol_col], label=vol_col, alpha=0.7, color='orange')
        
        ax1.set_title('比特币成交量')
        ax1.set_xlabel('日期/时间')
        ax1.set_ylabel('成交量')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
    
    # 如果只有数值列，绘制一般图表
    elif len(columns) >= 2:
        ax1 = fig.add_subplot(111)
        ax1.plot(df_sql.iloc[:, 0], df_sql.iloc[:, 1:], linewidth=2)
        ax1.set_title('数据可视化')
        ax1.set_xlabel(columns[0])
        ax1.set_ylabel('数值')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(columns[1:])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 以下是文件的其余部分，保持原样
# ====== arima_stock 工具类实现 ======
@register_tool('arima_stock')
class ARIMATool(BaseTool):
    """
    使用ARIMA模型对指定币子未来N天的价格进行预测
    """
    description = '使用ARIMA模型对指定币子未来N天的价格进行预测'
    parameters = [
        {
            'name': 'b_code',
            'type': 'string',
            'description': '币子代码，必填',
            'required': True
        },
        {
            'name': 'n',
            'type': 'integer',
            'description': '预测的天数',
            'default': 7
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import pandas as pd
        import numpy as np
        from statsmodels.tsa.arima.model import ARIMA
        import matplotlib.pyplot as plt
        import time
        import os
        from datetime import datetime, timedelta
        
        args = json.loads(params)
        b_code = args.get('b_code', 'BTC').strip().upper()
        n = args.get('n', 7)
        
        # 修正常见拼写错误并规范化交易对格式
        if b_code == 'BCT':
            b_code = 'BTC'
        symbol = f"{b_code}USDT"
        
        # 使用 Binance API 获取历史数据
        try:
            # 获取足够的历史数据，至少需要n*10天的数据来建立模型
            klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=n*10)
            
            if len(klines) < 30:  # 至少需要30天的数据
                return f"警告: 获取的历史数据不足30天，预测结果可能不准确。"
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                '开盘时间戳', '开盘价', '最高价', '最低价', '收盘价', '成交量', 
                '收盘时间戳', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'
            ])
            
            # 只保留收盘价并转换数据类型
            df['收盘价'] = df['收盘价'].astype(float)
            df['日期'] = pd.to_datetime(df['开盘时间戳'], unit='ms')
            
            # 设置日期为索引
            df.set_index('日期', inplace=True)
            
            # 使用ARIMA模型预测
            try:
                # 自动确定ARIMA参数（这里简化为(5,1,0)，实际应用中可以使用auto_arima）
                model = ARIMA(df['收盘价'], order=(5, 1, 0))
                model_fit = model.fit()
                
                # 预测未来n天的价格
                forecast = model_fit.forecast(steps=n)
                
                # 生成未来n天的日期索引
                last_date = df.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(n)]
                
                # 创建预测结果DataFrame
                forecast_df = pd.DataFrame({
                    '预测日期': future_dates,
                    '预测收盘价': forecast
                })
                
                # 格式化预测结果为表格
                forecast_table = forecast_df.to_markdown(index=False, tablefmt="pipe", 
                                                       headers=["预测日期", "预测收盘价(USDT)"])
                
                # 生成预测图表
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['收盘价'], label='历史收盘价', linewidth=2)
                plt.plot(future_dates, forecast, label='预测收盘价', color='red', linestyle='--', linewidth=2)
                plt.fill_between(future_dates, forecast * 0.95, forecast * 1.05, color='red', alpha=0.1, label='预测区间')
                plt.title(f'{b_code}未来{n}天价格预测 (ARIMA模型)')
                plt.xlabel('日期')
                plt.ylabel('价格 (USDT)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # 保存图表
                save_dir = os.path.join(os.path.dirname(__file__), 'btc_images')
                os.makedirs(save_dir, exist_ok=True)
                filename = f'btc_forecast_{int(time.time()*1000)}.png'
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 生成图表的markdown引用
                img_path = os.path.join('btc_images', filename)
                img_md = f'![{b_code}价格预测图]({img_path})'
                
                # 返回预测结果和图表
                return f"#{b_code}未来{n}天价格预测\n\n" \
                       f"## 预测结果\n{forecast_table}\n\n" \
                       f"## 预测图表\n{img_md}\n\n" \
                       f"## 预测说明\n" \
                       f"- 本预测基于ARIMA模型，使用最近{len(df)}天的历史数据\n" \
                       f"- 预测区间为±5%，实际价格可能在此区间内波动\n" \
                       f"- 加密货币市场波动较大，预测仅供参考，投资需谨慎\n" \
                       f"- 预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
            except Exception as model_error:
                # 如果ARIMA模型失败，使用简单的移动平均作为备选
                try:
                    # 计算7天移动平均
                    df['MA7'] = df['收盘价'].rolling(window=7).mean()
                    
                    # 使用最后一个MA7作为基准，添加随机波动进行预测
                    last_ma = df['MA7'].iloc[-1]
                    last_price = df['收盘价'].iloc[-1]
                    
                    # 计算历史波动率
                    df['returns'] = df['收盘价'].pct_change()
                    volatility = df['returns'].std()
                    
                    # 生成预测（基于历史趋势）
                    trend = (last_price / last_ma - 1) if last_ma > 0 else 0
                    forecast = [last_price * (1 + trend + np.random.normal(0, volatility)) for _ in range(n)]
                    
                    # 生成未来n天的日期索引
                    last_date = df.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(n)]
                    
                    # 创建预测结果DataFrame
                    forecast_df = pd.DataFrame({
                        '预测日期': future_dates,
                        '预测收盘价': forecast
                    })
                    
                    # 格式化预测结果为表格
                    forecast_table = forecast_df.to_markdown(index=False, tablefmt="pipe", 
                                                           headers=["预测日期", "预测收盘价(USDT)"])
                    
                    # 返回简化的预测结果
                    return f"#{b_code}未来{n}天价格预测\n\n" \
                           f"## 预测结果（简化模型）\n{forecast_table}\n\n" \
                           f"## 预测说明\n" \
                           f"- 由于ARIMA模型拟合失败，使用了基于历史趋势和波动率的简化模型\n" \
                           f"- 当前波动率: {volatility*100:.2f}%\n" \
                           f"- 当前价格趋势: {'上涨' if trend > 0 else '下跌'} {abs(trend)*100:.2f}%\n" \
                           f"- 加密货币市场波动较大，预测仅供参考，投资需谨慎\n" \
                           f"- 预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    
                except Exception as fallback_error:
                    return f"预测模型构建失败: {str(fallback_error)}"
        
        except Exception as e:
            return f"获取历史数据或构建预测模型失败: {str(e)}"


# ====== 优化的交易策略类 ======
class OptimizedTradingStrategy:
    def __init__(self):
        # 定义指标权重系统（可根据市场状态动态调整）
        self.trend_weights = {
            'MA': 0.25,      # 趋势跟踪权重
            'MACD': 0.20,    # 趋势动量权重
            'SAR': 0.15,     # 趋势反转权重
            'BOLL': 0.20,    # 波动率权重
            'RSI': 0.10,     # 超买超卖权重
            'KDJ': 0.10      # 短期动量权重
        }
        
        self.range_weights = {
            'RSI': 0.30,     # 震荡市中RSI更重要
            'BOLL': 0.25,    # 布林带在震荡市中很有效
            'KDJ': 0.20,     # KDJ适合震荡市
            'VOL': 0.15,     # 成交量确认
            'MA': 0.10       # 均线在震荡市中权重降低
        }

    def calculate_adx(self, df, period=14):
        """使用pandas实现ADX指标计算"""
        try:
            # 计算+DM和-DM
            df_copy = df.copy()
            df_copy['+DM'] = df_copy['最高价'].diff()
            df_copy['-DM'] = -df_copy['最低价'].diff()
            
            # 只保留大于零的值和另一个方向变化小于等于零的情况
            df_copy.loc[df_copy['+DM'] <= df_copy['-DM'] , '+DM'] = 0
            df_copy.loc[df_copy['-DM'] <= df_copy['+DM'] , '-DM'] = 0
            df_copy.loc[df_copy['+DM'] <= 0, '+DM'] = 0
            df_copy.loc[df_copy['-DM'] <= 0, '-DM'] = 0
            
            # 计算真实波幅(TR)
            df_copy['TR'] = np.maximum(
                df_copy['最高价'] - df_copy['最低价'],
                np.maximum(
                    abs(df_copy['最高价'] - df_copy['收盘价'].shift(1)),
                    abs(df_copy['最低价'] - df_copy['收盘价'].shift(1))
                )
            )
            
            # 计算14天的平滑TR、+DM和-DM
            df_copy['ATR'] = df_copy['TR'].rolling(window=period).mean()
            df_copy['+DM_Smooth'] = df_copy['+DM'].rolling(window=period).mean()
            df_copy['-DM_Smooth'] = df_copy['-DM'].rolling(window=period).mean()
            
            # 计算+DI和-DI
            df_copy['+DI'] = (df_copy['+DM_Smooth'] / df_copy['ATR']) * 100
            df_copy['-DI'] = (df_copy['-DM_Smooth'] / df_copy['ATR']) * 100
            
            # 计算DX
            df_copy['DX_Numerator'] = abs(df_copy['+DI'] - df_copy['-DI'])
            df_copy['DX_Denominator'] = df_copy['+DI'] + df_copy['-DI']
            df_copy['DX'] = (df_copy['DX_Numerator'] / df_copy['DX_Denominator']) * 100
            
            # 计算ADX
            df_copy['ADX'] = df_copy['DX'].rolling(window=period).mean()
            
            return df_copy['ADX'].values
        except Exception as e:
            print(f"ADX计算错误: {str(e)}")
            # 返回零值数组作为备用
            return np.zeros(len(df))

    def calculate_atr(self, df, period=14):
        """使用pandas实现ATR指标计算"""
        try:
            # 计算真实波幅(TR)
            df_copy = df.copy()
            df_copy['TR'] = np.maximum(
                df_copy['最高价'] - df_copy['最低价'],
                np.maximum(
                    abs(df_copy['最高价'] - df_copy['收盘价'].shift(1)),
                    abs(df_copy['最低价'] - df_copy['收盘价'].shift(1))
                )
            )
            
            # 使用平滑的ATR计算方法（类似talib的实现）
            df_copy['ATR'] = df_copy['TR'].rolling(window=period).mean()
            
            # 第一个ATR值之后使用平滑计算
            for i in range(period, len(df_copy)):
                df_copy.loc[df_copy.index[i], 'ATR'] = ((df_copy.loc[df_copy.index[i-1], 'ATR'] * (period-1)) + df_copy.loc[df_copy.index[i], 'TR']) / period
            
            return df_copy['ATR'].values
        except Exception as e:
            print(f"ATR计算错误: {str(e)}")
            # 备用计算方法
            tr = np.maximum(
                df['最高价'] - df['最低价'],
                np.maximum(
                    abs(df['最高价'] - df['收盘价'].shift(1)),
                    abs(df['最低价'] - df['收盘价'].shift(1))
                )
            )
            return tr.rolling(period).mean()

    def analyze_market_regime(self, df, adx_threshold=25):
        """分析市场状态：趋势市或震荡市"""
        if 'ADX' not in df.columns:
            df['ADX'] = self.calculate_adx(df)
        
        latest_adx = df['ADX'].iloc[-1]
        adx_avg = df['ADX'].tail(20).mean()
        
        # 判断市场状态
        if latest_adx > adx_threshold and adx_avg > adx_threshold:
            return 'trending'  # 趋势市
        else:
            return 'ranging'   # 震荡市

    def calculate_technical_score(self, df, current_price, market_regime):
        """计算技术指标综合得分"""
        latest = df.iloc[-1]
        scores = {}
        
        # MACD评分
        if latest['MACD'] > latest['Signal_Line'] and latest['MACD_Hist'] > 0:
            scores['MACD'] = 1.0
        elif latest['MACD'] < latest['Signal_Line'] and latest['MACD_Hist'] < 0:
            scores['MACD'] = -1.0
        else:
            scores['MACD'] = 0.0
        
        # RSI评分
        if latest['RSI'] < 30:
            scores['RSI'] = 1.0  # 超卖，看多
        elif latest['RSI'] > 70:
            scores['RSI'] = -1.0  # 超买，看空
        else:
            scores['RSI'] = 0.0
        
        # KDJ评分
        if latest['K'] > latest['D'] and latest['K'] < 80:
            scores['KDJ'] = 1.0
        elif latest['K'] < latest['D'] and latest['K'] > 20:
            scores['KDJ'] = -1.0
        else:
            scores['KDJ'] = 0.0
        
        # 移动平均线评分
        if latest['MA5'] > latest['MA10'] > latest['MA20']:
            scores['MA'] = 1.0
        elif latest['MA5'] < latest['MA10'] < latest['MA20']:
            scores['MA'] = -1.0
        else:
            scores['MA'] = 0.0
        
        # 布林带评分
        if current_price < latest['Lower_Band']:
            scores['BOLL'] = 1.0  # 触及下轨，可能反弹
        elif current_price > latest['Upper_Band']:
            scores['BOLL'] = -1.0  # 触及上轨，可能回调
        else:
            scores['BOLL'] = 0.0
        
        # SAR评分
        if current_price > latest['SAR']:
            scores['SAR'] = 1.0
        else:
            scores['SAR'] = -1.0
        
        # 成交量评分
        if latest['成交量'] > latest['VOL10'] * 1.2:
            # 成交量放大，加强当前趋势信号
            volume_strength = 0.5
        else:
            volume_strength = 0.0
        
        # 选择权重系统
        weights = self.trend_weights if market_regime == 'trending' else self.range_weights
        
        # 计算加权得分
        total_score = 0
        for indicator, score in scores.items():
            if indicator in weights:
                total_score += score * weights[indicator]
        
        # 加入成交量因素
        total_score += volume_strength * np.sign(total_score) if total_score != 0 else 0
        
        return total_score, scores

    def calculate_support_resistance(self, df):
        """计算支撑位和压力位"""
        # 使用多种方法计算支撑压力位
        recent_low = df['最低价'].tail(30).min()
        recent_high = df['最高价'].tail(30).max()
        
        # 方法1：基于近期高低点
        pivot = (recent_high + recent_low + df['收盘价'].iloc[-1]) / 3
        resistance1 = 2 * pivot - recent_low
        support1 = 2 * pivot - recent_high
        resistance2 = pivot + (recent_high - recent_low)
        support2 = pivot - (recent_high - recent_low)
        
        # 方法2：基于移动平均线
        ma20 = df['收盘价'].tail(20).mean()
        ma50 = df['收盘价'].tail(50).mean()
        
        # 综合两种方法
        support_levels = [
            round(min(support1, support2, ma20, ma50), 2),
            round(recent_low, 2)
        ]
        resistance_levels = [
            round(max(resistance1, resistance2, ma20, ma50), 2),
            round(recent_high, 2)
        ]
        
        return sorted(support_levels), sorted(resistance_levels, reverse=True)

    def analyze_trading_strategy(self, df, real_time_data):
        """
        优化的交易策略分析方法
        """
        try:
            current_price = real_time_data['current_price']
            
            # 初始化策略分析结果
            strategy = {
                '方向判断': '震荡',
                '建议操作': '观望',
                '市场状态': '未知',
                '综合得分': 0,
                '信号强度': '弱',
                '支撑位1': 0,
                '支撑位2': 0,
                '压力位1': 0,
                '压力位2': 0,
                '止损价格': 0,
                '止盈价格': 0,
                '风险收益比': 1.0,
                '仓位建议': '轻仓',
                '置信度': 0.0
            }
            
            # 分析市场状态
            market_regime = self.analyze_market_regime(df)
            strategy['市场状态'] = '趋势市' if market_regime == 'trending' else '震荡市'
            
            # 计算技术指标综合得分
            total_score, individual_scores = self.calculate_technical_score(
                df, current_price, market_regime
            )
            strategy['综合得分'] = round(total_score, 2)
            
            # 计算支撑位和压力位
            support_levels, resistance_levels = self.calculate_support_resistance(df)
            strategy['支撑位1'], strategy['支撑位2'] = support_levels[:2]
            strategy['压力位1'], strategy['压力位2'] = resistance_levels[:2]
            
            # 计算ATR用于风险管理
            if 'ATR' not in df.columns:
                df['ATR'] = self.calculate_atr(df)
            atr = df['ATR'].iloc[-1]
            
            # 根据得分和信号强度制定策略
            signal_strength = abs(total_score)
            
            if signal_strength > 0.7:
                strategy['信号强度'] = '强'
                strategy['仓位建议'] = '重仓'
                strategy['置信度'] = 0.8
            elif signal_strength > 0.3:
                strategy['信号强度'] = '中'
                strategy['仓位建议'] = '中仓'
                strategy['置信度'] = 0.6
            else:
                strategy['信号强度'] = '弱'
                strategy['仓位建议'] = '轻仓'
                strategy['置信度'] = 0.4
            
            # 制定交易决策
            if total_score > 0.5:  # 强烈看多
                strategy['方向判断'] = '强势上涨'
                strategy['建议操作'] = '买入'
                strategy['止损价格'] = round(current_price - 2 * atr, 2)
                strategy['止盈价格'] = round(current_price + 3 * atr, 2)
                strategy['风险收益比'] = 1.5
                
            elif total_score > 0.2:  # 温和看多
                strategy['方向判断'] = '温和上涨'
                strategy['建议操作'] = '买入'
                strategy['止损价格'] = round(current_price - 1.5 * atr, 2)
                strategy['止盈价格'] = round(current_price + 2 * atr, 2)
                strategy['风险收益比'] = 1.3
                
            elif total_score < -0.5:  # 强烈看空
                strategy['方向判断'] = '强势下跌'
                strategy['建议操作'] = '卖出'
                strategy['止损价格'] = round(current_price + 2 * atr, 2)
                strategy['止盈价格'] = round(current_price - 3 * atr, 2)
                strategy['风险收益比'] = 1.5
                
            elif total_score < -0.2:  # 温和看空
                strategy['方向判断'] = '温和下跌'
                strategy['建议操作'] = '卖出'
                strategy['止损价格'] = round(current_price + 1.5 * atr, 2)
                strategy['止盈价格'] = round(current_price - 2 * atr, 2)
                strategy['风险收益比'] = 1.3
                
            else:  # 震荡行情
                strategy['方向判断'] = '震荡整理'
                strategy['建议操作'] = '观望或区间操作'
                # 在震荡行情中，使用支撑压力位作为止损止盈
                strategy['止损价格'] = strategy['支撑位1']
                strategy['止盈价格'] = strategy['压力位1']
                profit_potential = strategy['压力位1'] - current_price
                loss_potential = current_price - strategy['支撑位1']
                if loss_potential > 0:
                    strategy['风险收益比'] = round(profit_potential / loss_potential, 2)
            
            # 添加详细的指标信号分析
            strategy['指标详情'] = individual_scores
            strategy['使用权重'] = self.trend_weights if market_regime == 'trending' else self.range_weights
            
            # 检查并处理NaN值，确保输出的价格都是有效数字
            import numpy as np
            if isinstance(strategy['止损价格'], float) and np.isnan(strategy['止损价格']):
                # 根据操作建议选择合适的默认止损价格
                if strategy['建议操作'] == '卖出':
                    strategy['止损价格'] = round(strategy['压力位1'], 2)  # 卖出时止损在压力位
                else:
                    strategy['止损价格'] = round(strategy['支撑位1'], 2)  # 买入时止损在支撑位
            
            if isinstance(strategy['止盈价格'], float) and np.isnan(strategy['止盈价格']):
                # 根据操作建议选择合适的默认止盈价格
                if strategy['建议操作'] == '卖出':
                    strategy['止盈价格'] = round(strategy['支撑位1'], 2)  # 卖出时止盈在支撑位
                else:
                    strategy['止盈价格'] = round(strategy['压力位1'], 2)  # 买入时止盈在压力位
            
            return strategy
            
        except Exception as e:
            raise Exception(f"分析交易策略失败: {str(e)}")

    def format_trading_strategy(self, strategy):
        """
        格式化交易策略结果
        """
        try:
            # 检查并处理NaN值，确保显示的价格都是有效数字
            import numpy as np
            stop_loss = strategy['止损价格']
            take_profit = strategy['止盈价格']
            
            # 根据操作建议处理止损止盈价格
            if strategy['建议操作'] == '卖出':
                # 对于卖出信号，止损应该在当前价格上方，止盈应该在当前价格下方
                if isinstance(stop_loss, float) and (np.isnan(stop_loss) or stop_loss == 0):
                    stop_loss = round(strategy['压力位1'], 2)  # 卖出时止损在压力位
                if isinstance(take_profit, float) and (np.isnan(take_profit) or take_profit == 0):
                    take_profit = round(strategy['支撑位1'], 2)  # 卖出时止盈在支撑位
            else:
                # 对于买入或其他信号，保持原有逻辑
                if isinstance(stop_loss, float) and (np.isnan(stop_loss) or stop_loss == 0):
                    stop_loss = round(strategy['支撑位1'], 2)
                if isinstance(take_profit, float) and (np.isnan(take_profit) or take_profit == 0):
                    take_profit = round(strategy['压力位1'], 2)
            
            formatted = f"""
📊 **交易策略分析报告**

**市场状态**: {strategy['市场状态']}
**方向判断**: {strategy['方向判断']} (得分: {strategy['综合得分']})
**信号强度**: {strategy['信号强度']} (置信度: {strategy['置信度']:.0%})

🎯 **操作建议**
- **主要操作**: {strategy['建议操作']}
- **仓位管理**: {strategy['仓位建议']}

💰 **关键价位**
- **支撑位**: {strategy['支撑位1']} / {strategy['支撑位2']}
- **压力位**: {strategy['压力位1']} / {strategy['压力位2']}
- **止损价格**: {stop_loss}
- **止盈价格**: {take_profit}

⚖️ **风险控制**
- **风险收益比**: 1:{strategy['风险收益比']}
- **建议仓位**: {strategy['仓位建议']}

📈 **技术指标信号**
"""
            # 添加各个指标的信号详情
            for indicator, signal in strategy.get('指标详情', {}).items():
                signal_text = "看多" if signal > 0 else "看空" if signal < 0 else "中性"
                formatted += f"- {indicator}: {signal_text} ({signal:+.1f})\n"
            
            # 添加权重信息
            formatted += f"\n⚖️ **使用的权重系统**\n"
            for indicator, weight in strategy.get('使用权重', {}).items():
                formatted += f"- {indicator}: {weight:.0%}\n"
            
            return formatted
            
        except Exception as e:
            return f"格式化策略结果时出错: {str(e)}"

    def optimize_parameters_based_on_regime(self, df, market_regime):
        """
        根据市场状态优化指标参数
        """
        optimized_params = {}
        
        if market_regime == 'trending':
            # 趋势市参数：更长的周期以减少假信号
            optimized_params.update({
                'ma_short': 10,    # 缩短均线捕捉趋势
                'ma_long': 30,
                'rsi_period': 14,
                'boll_period': 20
            })
        else:
            # 震荡市参数：更敏感的设置
            optimized_params.update({
                'ma_short': 5,     # 更短周期捕捉震荡
                'ma_long': 20,
                'rsi_period': 10,  # 更敏感的RSI
                'boll_period': 14
            })
        
        return optimized_params

# ====== get_real_time_price 工具类实现 ======
@register_tool('get_real_time_price')
class GetRealTimePriceTool(BaseTool, OptimizedTradingStrategy):
    """
    获取指定币子的实时价格数据，精确到秒
    """
    description = '获取指定币子的实时价格数据，精确到秒'
    parameters = [
        {
            'name': 'symbol',
            'type': 'string',
            'description': '交易对符号，如BTCUSDT，必填',
            'required': True
        }
    ]

    def __init__(self):
        BaseTool.__init__(self)
        OptimizedTradingStrategy.__init__(self)

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        symbol = args.get('symbol', 'BTCUSDT').strip().upper()  # 确保交易对符号为大写
        
        try:
            # 修正常见拼写错误
            if symbol == 'BCT':
                symbol = 'BTCUSDT'
            # 确保交易对符合Binance格式
            if 'USDT' not in symbol:
                symbol = f"{symbol}USDT"
            
            # 获取实时价格数据 - 添加额外的异常捕获
            try:
                real_time_data = self.fetch_real_time_price(symbol)
            except Exception as fetch_error:
                fetch_error_msg = str(fetch_error)
                # 处理fetch_real_time_price中抛出的特定异常
                if 'Invalid symbol' in fetch_error_msg:
                    return f"交易对符号错误: {symbol}。请使用正确的交易对格式，如'BTCUSDT'。"
                elif 'Connection' in fetch_error_msg or 'timed out' in fetch_error_msg:
                    return f"网络连接错误: 无法连接到交易所服务器。请检查您的网络连接。"
                else:
                    return f"获取实时价格数据失败: {fetch_error_msg}"
            
            # 双重验证数据结构 - 确保real_time_data是字典且包含current_price
            if not isinstance(real_time_data, dict):
                return f"获取实时价格时数据结构错误: 返回的数据类型不是字典。请检查网络连接或稍后重试。"
            
            if 'current_price' not in real_time_data:
                return f"获取实时价格时数据结构错误: 返回的字典中缺少current_price字段。请检查网络连接或稍后重试。"
            
            # 验证current_price的值是否有效
            if real_time_data['current_price'] == 0 or real_time_data['current_price'] is None:
                return f"获取实时价格失败: 当前价格为零或无效。可能是交易所API暂时不可用，请稍后重试。"
            
            # 获取最近的K线数据用于短期分析 - 添加异常捕获
            try:
                recent_klines = self.fetch_recent_klines(symbol)
            except Exception as kline_error:
                # 即使K线数据获取失败，也尝试继续，只返回价格信息而不显示图表
                price_table = self.format_real_time_price(real_time_data)
                return f"#{symbol}实时价格数据（精确到秒）\n\n" \
                       f"## 当前价格信息\n{price_table}\n\n" \
                       f"## 价格走势图表\n*注: 无法获取K线数据，因此无法显示价格走势图。*\n\n" \
                       f"*数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}*"
            
            # 生成可视化图表 - 添加异常捕获
            try:
                save_dir = os.path.join(os.path.dirname(__file__), 'btc_images')
                os.makedirs(save_dir, exist_ok=True)
                filename = f'btc_real_time_price_{int(time.time()*1000)}.png'
                save_path = os.path.join(save_dir, filename)
                
                self.plot_real_time_price(real_time_data, recent_klines, save_path, symbol)
                
                # 格式化实时数据为表格
                price_table = self.format_real_time_price(real_time_data)
                
                img_path = os.path.join('btc_images', filename)
                img_md = f'![{symbol}实时价格图表]({img_path})'
                
                # ===== 新增交易策略分析部分 =====
                trading_strategy_md = """
## 短期交易策略分析
"""
                
                try:
                    # 获取30天历史数据
                    historical_data = self.fetch_60day_historical_data(symbol)
                    
                    # 计算技术指标
                    historical_data_with_indicators = self.calculate_technical_indicators(historical_data)
                    
                    # 分析交易策略（使用优化的策略）
                    trading_strategy = self.analyze_trading_strategy(historical_data_with_indicators, real_time_data)
                    
                    # 格式化交易策略（使用优化的格式化方法）
                    formatted_strategy = self.format_trading_strategy(trading_strategy)
                    
                    # 生成技术指标图表
                    indicators_filename = f'btc_technical_indicators_{int(time.time()*1000)}.png'
                    indicators_save_path = os.path.join(save_dir, indicators_filename)
                    self.plot_technical_indicators(historical_data_with_indicators, trading_strategy, indicators_save_path, symbol)
                    
                    indicators_img_path = os.path.join('btc_images', indicators_filename)
                    indicators_img_md = f'![{symbol}技术指标图表]({indicators_img_path})'
                    
                    trading_strategy_md = f"""
## 短期交易策略分析

### 技术指标分析
{indicators_img_md}

### 交易策略建议
{formatted_strategy}

### 策略解读
根据优化的多指标综合分析系统，当前市场状态为**{trading_strategy['市场状态']}**，整体趋势判断为**{trading_strategy['方向判断']}**，建议**{trading_strategy['建议操作']}**。

- **信号强度**: {trading_strategy['信号强度']}（置信度: {trading_strategy['置信度']:.0%}）
- **支撑位和压力位**: 当前价格处于支撑位{trading_strategy['支撑位1']}和压力位{trading_strategy['压力位1']}之间
- **止损设置**: 建议将止损设置在{trading_strategy['止损价格']}，控制风险
- **止盈目标**: 建议将止盈设置在{trading_strategy['止盈价格']}
- **风险收益比**: 当前风险收益比为1:{trading_strategy['风险收益比']}
- **仓位建议**: {trading_strategy['仓位建议']}

请注意，加密货币市场波动较大，以上策略仅供参考，投资有风险，入市需谨慎。
                    """
                except Exception as strategy_error:
                    # 即使策略分析失败，也要确保返回基本价格信息
                    trading_strategy_md = f"""
## 短期交易策略分析
*注: 无法获取或分析交易策略数据: {str(strategy_error)}*
                    """
                
                # 构建返回结果，包含详细的实时价格数据和分析，供大模型进一步处理
                return f"#{symbol}实时价格数据与交易策略分析\n\n" \
                       f"## 当前价格信息\n{price_table}\n\n" \
                       f"## 价格走势图表\n{img_md}\n\n" \
                       f"{trading_strategy_md}\n\n" \
                       f"*数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}*"
                
            except Exception as plot_error:
                # 即使图表生成失败，也返回价格信息
                price_table = self.format_real_time_price(real_time_data)
                
                # 尝试获取交易策略分析（即使图表生成失败）
                trading_strategy_md = """
## 短期交易策略分析
*注: 无法生成图表，但尝试获取基本策略分析...*
"""
                
                try:
                    # 获取30天历史数据
                    historical_data = self.fetch_60day_historical_data(symbol)
                    historical_data_with_indicators = self.calculate_technical_indicators(historical_data)
                    trading_strategy = self.analyze_trading_strategy(historical_data_with_indicators, real_time_data)
                    formatted_strategy = self.format_trading_strategy(trading_strategy)
                    
                    trading_strategy_md = f"""
## 短期交易策略分析

### 交易策略建议
{formatted_strategy}

### 策略解读
根据优化的多指标综合分析系统，当前市场状态为**{trading_strategy['市场状态']}**，整体趋势判断为**{trading_strategy['方向判断']}**，建议**{trading_strategy['建议操作']}**。

- **信号强度**: {trading_strategy['信号强度']}（置信度: {trading_strategy['置信度']:.0%}）
- **支撑位和压力位**: 当前价格处于支撑位{trading_strategy['支撑位1']}和压力位{trading_strategy['压力位1']}之间
- **止损设置**: 建议将止损设置在{trading_strategy['止损价格']}，控制风险
- **止盈目标**: 建议将止盈设置在{trading_strategy['止盈价格']}
- **风险收益比**: 当前风险收益比为1:{trading_strategy['风险收益比']}
- **仓位建议**: {trading_strategy['仓位建议']}
                    """
                except:
                    pass
                
                # 打印错误信息以便调试
                print(f"图表生成错误: {str(plot_error)}")
                
                # 确保btc_images目录存在
                save_dir = os.path.join(os.path.dirname(__file__), 'btc_images')
                os.makedirs(save_dir, exist_ok=True)
                
                return f"#{symbol}实时价格数据（精确到秒）\n\n" \
                       f"## 当前价格信息\n{price_table}\n\n" \
                       f"## 价格走势图表\n*注: 图表生成失败，但已获取到价格数据。错误: {str(plot_error)}*\n\n" \
                       f"{trading_strategy_md}\n\n" \
                       f"*数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}*"
            
        except Exception as e:
            return f"获取实时价格数据时发生错误: {str(e)}"
    
    def fetch_real_time_price(self, symbol):
        """
        从Binance API获取实时价格数据
        """
        try:
            # 获取最新价格
            ticker = client.get_ticker(symbol=symbol)
            
            # 获取订单簿深度数据
            order_book = client.get_order_book(symbol=symbol, limit=1)
            
            # 构建返回数据结构
            real_time_data = {
                'symbol': symbol,
                'current_price': float(ticker['lastPrice']),
                'bid_price': float(order_book['bids'][0][0]) if order_book['bids'] else 0,
                'ask_price': float(order_book['asks'][0][0]) if order_book['asks'] else 0,
                'bid_quantity': float(order_book['bids'][0][1]) if order_book['bids'] else 0,
                'ask_quantity': float(order_book['asks'][0][1]) if order_book['asks'] else 0,
                'price_change_24h': float(ticker['priceChange']),
                'price_change_percent_24h': float(ticker['priceChangePercent']),
                'high_price_24h': float(ticker['highPrice']),
                'low_price_24h': float(ticker['lowPrice']),
                'volume_24h': float(ticker['volume']),
                'last_trade_time': datetime.now()
            }
            
            return real_time_data
        except Exception as e:
            # 提供更具体的错误信息
            if 'Invalid symbol' in str(e):
                raise ValueError(f"无效的交易对: {symbol}")
            elif 'Connection' in str(e):
                raise ConnectionError("网络连接失败，请检查您的网络连接")
            else:
                raise Exception(f"获取实时价格数据时出错: {str(e)}")
    
    def fetch_recent_klines(self, symbol, limit=100, interval=Client.KLINE_INTERVAL_15MINUTE):
        """
        获取最近的K线数据用于绘制短期走势图
        """
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                '开盘时间戳', '开盘价', '最高价', '最低价', '收盘价', '成交量',
                '收盘时间戳', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'
            ])
            
            # 转换数据类型和时间戳
            df['开盘时间'] = pd.to_datetime(df['开盘时间戳'], unit='ms')
            df['开盘价'] = df['开盘价'].astype(float)
            df['收盘价'] = df['收盘价'].astype(float)
            df['最高价'] = df['最高价'].astype(float)
            df['最低价'] = df['最低价'].astype(float)
            df['成交量'] = df['成交量'].astype(float)
            
            return df
        except Exception as e:
            raise Exception(f"获取K线数据失败: {str(e)}")
    
    def fetch_60day_historical_data(self, symbol):
        """
        获取近30天的历史数据，用于计算技术指标
        """
        try:
            # 获取30天的1小时K线数据（30天 * 24小时 = 720个数据点）
            klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=1440)
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                '开盘时间戳', '开盘价', '最高价', '最低价', '收盘价', '成交量',
                '收盘时间戳', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'
            ])
            
            # 转换数据类型和时间戳
            df['时间'] = pd.to_datetime(df['开盘时间戳'], unit='ms')
            df['开盘价'] = df['开盘价'].astype(float)
            df['收盘价'] = df['收盘价'].astype(float)
            df['最高价'] = df['最高价'].astype(float)
            df['最低价'] = df['最低价'].astype(float)
            df['成交量'] = df['成交量'].astype(float)
            
            return df
        except Exception as e:
            raise Exception(f"获取历史数据失败: {str(e)}")
    
    def calculate_technical_indicators(self, df):
        """
        计算各种技术指标
        """
        try:
            # 计算MA (移动平均线)
            df['MA5'] = df['收盘价'].rolling(window=5).mean()
            df['MA10'] = df['收盘价'].rolling(window=10).mean()
            df['MA20'] = df['收盘价'].rolling(window=20).mean()
            df['MA60'] = df['收盘价'].rolling(window=60).mean()
            
            # 计算RSI (相对强弱指标)
            delta = df['收盘价'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算KDJ指标
            n = 9
            m1 = 3
            m2 = 3
            
            # 计算RSV值
            df['LLV'] = df['最低价'].rolling(window=n).min()
            df['HHV'] = df['最高价'].rolling(window=n).max()
            df['RSV'] = (df['收盘价'] - df['LLV']) / (df['HHV'] - df['LLV']) * 100
            
            # 计算K、D、J值
            df['K'] = df['RSV'].ewm(alpha=1/m1, adjust=False).mean()
            df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            
            # 计算MACD (移动平均收敛散度)
            exp1 = df['收盘价'].ewm(span=12, adjust=False).mean()
            exp2 = df['收盘价'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # 计算BOLL (布林带)
            df['MA20'] = df['收盘价'].rolling(window=20).mean()
            df['STD20'] = df['收盘价'].rolling(window=20).std()
            df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
            df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
            
            # 计算SAR (抛物线转向指标)
            df['SAR'] = 0.0
            af = 0.02
            max_af = 0.2
            sar = df['收盘价'].iloc[0]
            ep = df['收盘价'].iloc[0]
            trend = 1  # 1表示上升趋势，-1表示下降趋势
            
            for i in range(1, len(df)):
                if trend == 1:
                    sar = sar + af * (ep - sar)
                    if df['最低价'].iloc[i] < sar:
                        trend = -1
                        sar = ep
                        ep = df['最低价'].iloc[i]
                        af = 0.02
                    else:
                        if df['最高价'].iloc[i] > ep:
                            ep = df['最高价'].iloc[i]
                            af = min(af + 0.02, max_af)
                else:
                    sar = sar + af * (ep - sar)
                    if df['最高价'].iloc[i] > sar:
                        trend = 1
                        sar = ep
                        ep = df['最高价'].iloc[i]
                        af = 0.02
                    else:
                        if df['最低价'].iloc[i] < ep:
                            ep = df['最低价'].iloc[i]
                            af = min(af + 0.02, max_af)
                df.loc[df.index[i], 'SAR'] = sar
            
            # 计算VOL (成交量)
            df['VOL5'] = df['成交量'].rolling(window=5).mean()
            df['VOL10'] = df['成交量'].rolling(window=10).mean()
            
            # 计算OBV (能量潮指标)
            df['OBV'] = 0
            for i in range(1, len(df)):
                if df['收盘价'].iloc[i] > df['收盘价'].iloc[i-1]:
                    df.loc[df.index[i], 'OBV'] = df['OBV'].iloc[i-1] + df['成交量'].iloc[i]
                elif df['收盘价'].iloc[i] < df['收盘价'].iloc[i-1]:
                    df.loc[df.index[i], 'OBV'] = df['OBV'].iloc[i-1] - df['成交量'].iloc[i]
                else:
                    df.loc[df.index[i], 'OBV'] = df['OBV'].iloc[i-1]
            
            # OptimizedTradingStrategy类已经包含了自己的ADX和ATR计算方法
        # 这里不需要提前计算这些指标，会在分析策略时自动计算
            
            return df
        except Exception as e:
            raise Exception(f"计算技术指标失败: {str(e)}")
    
    def plot_real_time_price(self, real_time_data, recent_klines, save_path, symbol):
        """
        绘制实时价格走势图
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制K线的收盘价
            plt.plot(recent_klines['开盘时间'], recent_klines['收盘价'], linewidth=2, label='收盘价')
            
            # 标记当前价格
            current_price = real_time_data['current_price']
            last_time = recent_klines['开盘时间'].iloc[-1]
            plt.scatter(last_time, current_price, color='red', s=100, zorder=5, label=f'当前价格: {current_price}')
            
            # 添加价格变化信息
            price_change_percent = real_time_data['price_change_percent_24h']
            change_color = 'green' if price_change_percent > 0 else 'red'
            change_text = f"24h变化: {'+' if price_change_percent > 0 else ''}{price_change_percent:.2f}%"
            
            # 添加标题和标签
            plt.title(f'{symbol} 实时价格走势图\n{change_text}', color=change_color)
            plt.xlabel('时间')
            plt.ylabel('价格 (USDT)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 优化x轴时间显示
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            raise Exception(f"绘制实时价格图表失败: {str(e)}")
    
    def plot_technical_indicators(self, df, strategy, save_path, symbol):
        """
        绘制技术指标图表
        """
        try:
            # 创建一个包含多个子图的图表
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            # 1. 价格和移动平均线
            ax1 = axes[0]
            ax1.plot(df['时间'], df['收盘价'], label='收盘价', linewidth=2)
            ax1.plot(df['时间'], df['MA5'], label='MA5', linewidth=1, alpha=0.7)
            ax1.plot(df['时间'], df['MA10'], label='MA10', linewidth=1, alpha=0.7)
            ax1.plot(df['时间'], df['MA20'], label='MA20', linewidth=1, alpha=0.7)
            ax1.plot(df['时间'], df['SAR'], '^g' if df['收盘价'].iloc[-1] > df['SAR'].iloc[-1] else 'vr', markersize=3, label='SAR')
            
            # 添加布林带
            ax1.plot(df['时间'], df['Upper_Band'], '--', color='gray', alpha=0.5, label='布林带上轨')
            ax1.plot(df['时间'], df['Lower_Band'], '--', color='gray', alpha=0.5, label='布林带下轨')
            ax1.fill_between(df['时间'], df['Upper_Band'], df['Lower_Band'], color='gray', alpha=0.1)
            
            # 添加支撑位和压力位
            ax1.axhline(y=strategy['支撑位1'], color='green', linestyle='--', alpha=0.7, label=f'支撑位1: {strategy["支撑位1"]}')
            ax1.axhline(y=strategy['支撑位2'], color='lightgreen', linestyle='--', alpha=0.5, label=f'支撑位2: {strategy["支撑位2"]}')
            ax1.axhline(y=strategy['压力位1'], color='red', linestyle='--', alpha=0.7, label=f'压力位1: {strategy["压力位1"]}')
            ax1.axhline(y=strategy['压力位2'], color='pink', linestyle='--', alpha=0.5, label=f'压力位2: {strategy["压力位2"]}')
            
            ax1.set_title(f'{symbol} 价格与技术指标分析')
            ax1.set_ylabel('价格 (USDT)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left')
            
            # 2. RSI指标
            ax2 = axes[1]
            ax2.plot(df['时间'], df['RSI'], label='RSI', linewidth=2, color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='中性线(50)')
            ax2.set_ylabel('RSI')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper left')
            
            # 3. MACD指标
            ax3 = axes[2]
            ax3.plot(df['时间'], df['MACD'], label='MACD', linewidth=2, color='blue')
            ax3.plot(df['时间'], df['Signal_Line'], label='信号线', linewidth=2, color='orange')
            ax3.bar(df['时间'], df['MACD_Hist'], label='MACD柱状', color=['green' if x > 0 else 'red' for x in df['MACD_Hist']], alpha=0.7)
            ax3.set_ylabel('MACD')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend(loc='upper left')
            
            # 4. KDJ指标
            ax4 = axes[3]
            ax4.plot(df['时间'], df['K'], label='K线', linewidth=1.5, color='blue')
            ax4.plot(df['时间'], df['D'], label='D线', linewidth=1.5, color='orange')
            ax4.plot(df['时间'], df['J'], label='J线', linewidth=1.5, color='green')
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='超买线(80)')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='超卖线(20)')
            ax4.set_ylabel('KDJ')
            ax4.set_xlabel('时间')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend(loc='upper left')
            
            # 优化x轴时间显示
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            raise Exception(f"绘制技术指标图表失败: {str(e)}")
    
    def format_real_time_price(self, real_time_data):
        """
        格式化实时价格数据为Markdown表格
        """
        try:
            # 格式化价格变化，添加正负号
            price_change = real_time_data['price_change_24h']
            price_change_percent = real_time_data['price_change_percent_24h']
            change_sign = '+' if price_change > 0 else ''
            change_percent_sign = '+' if price_change_percent > 0 else ''
            
            # 创建表格
            table = f"""
| 指标 | 值 |
|------|------|
| 当前价格 | {real_time_data['current_price']} USDT |
| 买一价 | {real_time_data['bid_price']} USDT |
| 卖一价 | {real_time_data['ask_price']} USDT |
| 24小时涨跌幅 | {change_sign}{price_change} USDT ({change_percent_sign}{price_change_percent}%) |
| 24小时最高价 | {real_time_data['high_price_24h']} USDT |
| 24小时最低价 | {real_time_data['low_price_24h']} USDT |
| 24小时成交量 | {real_time_data['volume_24h']} {real_time_data['symbol'].replace('USDT', '')} |
"""
            
            return table.strip()
        except Exception as e:
            raise Exception(f"格式化实时价格数据失败: {str(e)}")

# ====== 获取LLM配置的函数 ======
def get_llm_cfg():
    """配置LLM模型参数"""
    llm_cfg = {
        # 使用 DashScope 提供的模型服务：
        'model': 'qwen-turbo',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
        'generate_cfg': {
            'top_p': 0.8
        }
    }
    return llm_cfg

# ====== 初始化比特币助手服务 ======
def init_agent_service():
    """
    初始化比特币价格分析助手服务
    """
    try:
        # 创建助手实例
        bot = Assistant(
            llm=get_llm_cfg(),
            name='比特币分析助手',
            description='比特币价格数据查询、实时价格和预测分析',
            system_message=system_prompt,
            # 包含所有需要的工具实例
            function_list=[ExcSQLTool(), ARIMATool(), GetRealTimePriceTool()],
        )
        print("比特币价格分析助手初始化成功！")
        print("已启用功能：")
        print("1. SQL查询与数据可视化")
        print("2. ARIMA模型价格预测")
        print("3. 实时价格数据获取与分析")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_gui():
    """
    启动Web图形界面模式
    """
    try:
        bot = init_agent_service()
        
        chatbot_config = {
            'title': '比特币价格分析助手',
            'description': '提供实时比特币价格、技术指标分析和交易策略建议',
            'prompt.suggestions': [
                '查询2023年比特币的最高价和最低价',
                '分析最近3个月比特币价格的走势',
                '对比特币的成交量进行月度统计并分析',
                '使用ARIMA模型预测比特币未来7天的价格',
                '预测BTCUSDT未来14天的价格趋势',
                '获取BTCUSDT的实时价格并分析短期走势',
                '查看比特币的最新价格、技术指标和投资建议'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        print("访问 http://127.0.0.1:7861 开始使用比特币价格分析助手")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


def main():
    """主函数，提供终端和Web界面两种模式"""
    print("比特币价格分析助手启动中...")
    choice = 2  # 默认启动Web图形界面模式
    try:
        if choice == 1:
            print("启动终端交互模式...")
            print("终端模式暂未实现")
        else:
            print("启动Web图形界面模式...")
            app_gui()
    except KeyboardInterrupt:
        print("\n程序被用户中断，退出...")
    except Exception as e:
        print(f"程序运行时出错: {str(e)}")


if __name__ == '__main__':
    main()