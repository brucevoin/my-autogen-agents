import yfinance as yf

# Fetch the latest news from Yahoo Finance
news = yf.Ticker('AAPL').news

# Print the news headlines
for item in news:
    print(item['title'])