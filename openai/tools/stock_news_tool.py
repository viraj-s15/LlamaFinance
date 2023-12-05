from langchain.tools import tool

import sys

sys.path.insert(1, "../args/")

from args import CompanyNewsArgs


@tool("company_news", args_schema=CompanyNewsArgs)
def get_company_news(stock_name: str) -> str:
    """
    Function which gets news about the company
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    res = ticker.news
    output = ""
    for i in res:
        output += i["title"] + i["publisher"] + i["link"] + "\n"
    return output
