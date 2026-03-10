import streamlit as st
import efinance as ef
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import io

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(stock_code, days=60):
    try:
        df = ef.stock.get_quote_history(stock_code)
        if df is not None and len(df) > 0:
            stock_name = None
            if '股票名称' in df.columns:
                stock_name = df['股票名称'].iloc[0]
            elif 'name' in df.columns:
                stock_name = df['name'].iloc[0]
            
            df = df.tail(days)
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.reset_index(drop=True)
            
            if stock_name:
                df['股票名称'] = stock_name
            
            return df, stock_name
        return None, None
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None, None

def calculate_5day_cycles(df):
    if len(df) < 5:
        return pd.DataFrame()
    
    cycles = []
    for i in range(4, len(df)):
        window = df.iloc[i-4:i+1]
        
        price_change = ((window['收盘'].iloc[-1] - window['收盘'].iloc[0]) / window['收盘'].iloc[0]) * 100
        
        up_days = 0
        down_days = 0
        flat_days = 0
        
        for j in range(1, 5):
            if window['收盘'].iloc[j] > window['收盘'].iloc[j-1]:
                up_days += 1
            elif window['收盘'].iloc[j] < window['收盘'].iloc[j-1]:
                down_days += 1
            else:
                flat_days += 1
        
        cycles.append({
            '日期': window['日期'].iloc[-1],
            '累计涨跌幅': price_change,
            '上涨天数': up_days,
            '下跌天数': down_days,
            '平盘天数': flat_days
        })
    
    return pd.DataFrame(cycles)

def identify_trend(df_stats):
    if len(df_stats) < 10:
        return '震荡趋势'
    
    recent = df_stats['累计涨跌幅'].tail(10)
    
    if recent.mean() > 3 and recent.tail(5).mean() > recent.head(5).mean():
        return '上涨趋势'
    elif recent.mean() < -3 and recent.tail(5).mean() < recent.head(5).mean():
        return '下跌趋势'
    else:
        return '震荡趋势'

def identify_signals(df_stats):
    signals = {'buy': [], 'sell': []}
    cumulative_change = df_stats['累计涨跌幅']
    
    trend_type = identify_trend(df_stats)
    
    if trend_type == '震荡趋势':
        for i in range(1, len(df_stats)):
            if cumulative_change.iloc[i-1] < -5 and cumulative_change.iloc[i] > cumulative_change.iloc[i-1]:
                signals['buy'].append(i)
            if cumulative_change.iloc[i-1] > 5 and cumulative_change.iloc[i] < 0:
                signals['sell'].append(i)
    else:
        for i in range(1, len(df_stats)):
            if cumulative_change.iloc[i-1] < 0 and cumulative_change.iloc[i] >= 0:
                signals['buy'].append(i)
            if cumulative_change.iloc[i-1] > 5 and cumulative_change.iloc[i] < 0:
                signals['sell'].append(i)
    
    return signals, trend_type

def plot_5day_trend(df, df_stats, signals, trend_type, stock_code, stock_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    dates = df_stats['日期']
    cumulative_change = df_stats['累计涨跌幅']
    x = range(len(dates))
    
    ax1.plot(x, cumulative_change, color='blue', linewidth=1.5, label='累计涨跌幅')
    ax1.fill_between(x, cumulative_change, 0, where=(cumulative_change >= 0), color='red', alpha=0.3)
    ax1.fill_between(x, cumulative_change, 0, where=(cumulative_change < 0), color='green', alpha=0.3)
    
    for idx in signals['buy']:
        if idx < len(x):
            ax1.scatter(x[idx], cumulative_change.iloc[idx], color='red', s=100, marker='^', zorder=5, label='买点')
    
    for idx in signals['sell']:
        if idx < len(x):
            ax1.scatter(x[idx], cumulative_change.iloc[idx], color='green', s=100, marker='v', zorder=5, label='卖点')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    if stock_name:
        title = f'{stock_code} {stock_name} 5天周期累计涨跌幅趋势'
    else:
        title = f'{stock_code} 5天周期累计涨跌幅趋势'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('累计涨跌幅 (%)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    
    price_data = df['收盘'].iloc[4:].reset_index(drop=True)
    price_ma5 = price_data.rolling(window=5).mean()
    ax2.plot(range(len(price_ma5)), price_ma5, linewidth=2, color='orange', label='5日均线')
    ax2.set_ylabel('5日均线', fontsize=11, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_xlabel('时间', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig

def calculate_win_rate(df, signals, trend_type):
    if not signals['buy'] or not signals['sell']:
        return None
    
    buy_prices = []
    sell_prices = []
    buy_dates = []
    sell_dates = []
    
    for idx in signals['buy']:
        if idx + 4 < len(df):
            buy_prices.append(df['收盘'].iloc[idx + 4])
            buy_dates.append(df['日期'].iloc[idx + 4])
    
    for idx in signals['sell']:
        if idx + 4 < len(df):
            sell_prices.append(df['收盘'].iloc[idx + 4])
            sell_dates.append(df['日期'].iloc[idx + 4])
    
    if not buy_prices or not sell_prices:
        return None
    
    trades = []
    for i, (buy_price, buy_date) in enumerate(zip(buy_prices, buy_dates)):
        for j, (sell_price, sell_date) in enumerate(zip(sell_prices, sell_dates)):
            if sell_date > buy_date:
                profit = ((sell_price - buy_price) / buy_price) * 100
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit': profit
                })
                break
    
    if not trades:
        return None
    
    wins = sum(1 for t in trades if t['profit'] > 0)
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_profit = sum(t['profit'] for t in trades) / total if total > 0 else 0
    total_profit = sum(t['profit'] for t in trades)
    
    return {
        'total': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'total_profit': total_profit,
        'trades': trades
    }

st.set_page_config(page_title="股票5天周期分析工具", page_icon="📈", layout="wide")

st.title("📈 股票5天周期分析工具")
st.markdown("基于5天周期的股票买卖点分析系统")

with st.sidebar:
    st.header("参数设置")
    stock_code = st.text_input("股票代码", value="000001", help="输入股票代码，如：000001")
    days = st.slider("分析天数", 30, 250, 60)
    ma_period = st.selectbox("均线周期", [5, 10, 20, 30, 60], index=0)
    
    analyze_btn = st.button("开始分析", type="primary")

if analyze_btn:
    with st.spinner("正在获取股票数据..."):
        df, stock_name = get_stock_data(stock_code, days * 2)
        
        if df is not None and len(df) > 0:
            df_stats = calculate_5day_cycles(df)
            
            if len(df_stats) > 0:
                signals, trend_type = identify_signals(df_stats)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("股票代码", stock_code)
                with col2:
                    st.metric("股票名称", stock_name if stock_name else "未知")
                with col3:
                    st.metric("趋势类型", trend_type)
                
                st.subheader("📊 5天周期趋势图")
                fig = plot_5day_trend(df, df_stats, signals, trend_type, stock_code, stock_name)
                st.pyplot(fig)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔴 买点信号")
                    if signals['buy']:
                        for idx in signals['buy']:
                            if idx < len(df_stats):
                                st.write(f"📅 {df_stats['日期'].iloc[idx].strftime('%Y-%m-%d')}: 累计涨跌幅 {df_stats['累计涨跌幅'].iloc[idx]:.2f}%")
                    else:
                        st.info("暂无买点信号")
                
                with col2:
                    st.subheader("🟢 卖点信号")
                    if signals['sell']:
                        for idx in signals['sell']:
                            if idx < len(df_stats):
                                st.write(f"📅 {df_stats['日期'].iloc[idx].strftime('%Y-%m-%d')}: 累计涨跌幅 {df_stats['累计涨跌幅'].iloc[idx]:.2f}%")
                    else:
                        st.info("暂无卖点信号")
                
                st.subheader("📈 统计摘要")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_change = df_stats['累计涨跌幅'].mean()
                max_change = df_stats['累计涨跌幅'].max()
                min_change = df_stats['累计涨跌幅'].min()
                
                with col1:
                    st.metric("平均累计涨跌幅", f"{avg_change:.2f}%")
                with col2:
                    st.metric("最大累计涨跌幅",:.2f}% f"{max_change")
                with col3:
                    st.metric("最小累计涨跌幅", f"{min_change:.2f}%")
                with col4:
                    st.metric("平均上涨天数", f"{df_stats['上涨天数'].mean():.2f}")
                
                win_rate_data = calculate_win_rate(df, signals, trend_type)
                
                if win_rate_data:
                    st.subheader("🎯 交易胜率统计")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("总交易次数", win_rate_data['total'])
                    with col2:
                        st.metric("盈利次数", win_rate_data['wins'])
                    with col3:
                        st.metric("亏损次数", win_rate_data['losses'])
                    with col4:
                        st.metric("胜率", f"{win_rate_data['win_rate']:.2f}%")
                    with col5:
                        st.metric("总盈亏", f"{win_rate_data['total_profit']:.2f}%")
                    
                    st.subheader("📋 交易记录")
                    trade_df = pd.DataFrame(win_rate_data['trades'])
                    st.dataframe(trade_df, use_container_width=True)
                else:
                    st.warning("交易信号不足，无法计算胜率")
                
                st.subheader("💡 建议")
                if trend_type == "上涨趋势":
                    st.success("当前处于上涨趋势，建议顺势而为持股待涨，逢低买入")
                elif trend_type == "下跌趋势":
                    st.error("当前处于下跌趋势，建议观望为主，及时止损")
                else:
                    st.info("当前处于震荡趋势，建议高抛低吸，谨慎操作")
        else:
            st.error("无法获取股票数据，请检查股票代码是否正确")

st.markdown("---")
st.markdown("**使用说明**：输入股票代码，设置分析参数，点击开始分析即可查看股票的5天周期分析结果。")
