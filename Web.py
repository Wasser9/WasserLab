import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

def main():
    st.title("股票时间回归模型研究与可视化")
    st.write("仅限于学习用途，不构成任何投资建议。")

    # 用户输入股票代码及日期范围
    ticker = st.text_input("请输入股票代码（如：AAPL）", "AAPL")
    start_date = st.date_input("开始日期", datetime(2020, 1, 1))
    end_date = st.date_input("结束日期", datetime.today())

    if st.button("获取数据并分析"):
        # 下载股票数据，auto_adjust 默认为 True（自动调整数据）
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            st.error("未获取到数据，请检查股票代码或日期范围。")
            return

        st.subheader("股票数据预览")
        st.dataframe(df)

        # 数据预处理
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        # 新增列 'Day'，表示从第一天开始的天数
        df['Day'] = (df['Date'] - df['Date'].min()).dt.days

        # 构建线性回归模型：以天数为自变量，收盘价为因变量
        X = df[['Day']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        st.subheader("回归模型参数")
        st.write("截距：", model.intercept_)
        st.write("斜率：", model.coef_[0])

        # 美化回归图：使用 matplotlib 内置样式 'ggplot'
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label="Close price", color='blue', marker='o', linestyle='-', markersize=4)
        ax.plot(df['Date'], y_pred, label="Predict Close price", color='red', linestyle='--', linewidth=2)

        # 设置标题和坐标轴标签
        ax.set_title("Stock close price Regression chart", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Close", fontsize=14)
        ax.legend(fontsize=12)

        # 优化 X 轴显示：旋转日期标签、紧凑布局
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.subheader("Regression chart")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
