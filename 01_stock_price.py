import streamlit as st
import yfinance as yf

# print("First stream lit  program")

# markdown lang 
st.write("# Stock Price App")
st.write("Show are the Stock **closing price** and **volume** of Google")

tickerSymbol = 'GOOGL'
# get data of this ticker
tickerData = yf.Ticker(tickerSymbol)

tickerDF = tickerData.history(period='1d',start='2010-5-31',end = '2020-5-31')
# DF cols
# open  high  low  Close  volume  diviends  stock  splits
st.line_chart(tickerDF.Close)
st.line_chart(tickerDF.Volume)


