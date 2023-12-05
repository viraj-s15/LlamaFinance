from langchain.tools import Tool, tool
from pydantic import BaseModel, Field
import sys

sys.path.insert(1, "../args/")

from args import GetTickerArgs


@tool("ticker", args_schema=GetTickerArgs)
def get_ticker(company_name) -> str:
    """
    Provides the ticker for a given company name
    """
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={"User-Agent": user_agent})
    data = res.json()
    company_code = data["quotes"][0]["symbol"]
    return company_code


@tool("generic_stock_information", args_schema=GenericStockInfoArgs)
def get_generic_stock_information(stock_name: str) -> str:
    """
    Function which retrieves general information related to the stock, such as current price,etc
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    info = dict()
    info["Current Price"] = ticker.info["currentPrice"]
    info["Previous Close"] = ticker.info["previousClose"]
    info["Open Price"] = ticker.info["open"]
    info["Day Low"] = ticker.info["dayLow"]
    info["Day High"] = ticker.info["dayHigh"]
    info["Regular Market Previous Close"] = ticker.info["regularMarketPreviousClose"]
    info["Regular Market Previous Open"] = ticker.info["regularMarketOpen"]
    info["Regular Market Day Low"] = ticker.info["regularMarketDayLow"]
    info["Regular Market Day High"] = ticker.info["regularMarketDayHigh"]
    info["Market Cap"] = ticker.info["marketCap"]
    info["52 Week low"] = ticker.info["fiftyTwoWeekLow"]
    info["52 Week high"] = ticker.info["fiftyTwoWeekHigh"]
    info["Target High Price"] = ticker.info["targetHighPrice"]
    info["Target Low Price"] = ticker.info["targetLowPrice"]
    info["Target Mean Price"] = ticker.info["targetMeanPrice"]
    info["Target Median Price"] = ticker.info["targetMedianPrice"]

    return str(info)
