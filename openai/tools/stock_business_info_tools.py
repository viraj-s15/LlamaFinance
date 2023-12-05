import yfinance as yf
from args import BusinessInfoArgs, CompanyNewsArgs, ContactInfoArgs, OfficerInfoArgs
from langchain.tools import tool


@tool("company_news", args_schema=CompanyNewsArgs)
def get_company_news(stock_name: str) -> str:
    """
    Provides news about the company
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    res = ticker.news
    output = ""
    for i in res:
        output += i["title"] + i["publisher"] + i["link"] + "\n"
    return output


@tool("contact_information", args_schema=ContactInfoArgs)
def get_contact_information(stock_name: str) -> str:
    """
    Provides contact information of the company
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    info = dict()
    info["Address"] = ticker.info["address1"]
    info["City"] = ticker.info["city"]
    info["State"] = ticker.info["state"]
    info["Zip Code"] = ticker.info["zip"]
    info["Phone Number"] = ticker.info["phone"]
    info["Website"] = ticker.info["website"]
    return str(info)


@tool("business_info", args_schema=BusinessInfoArgs)
def get_business_info(stock_name: str) -> str:
    """
    Provides general data of its stock including current price,open,close,etc
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    info = dict()
    info["Industry"] = ticker.info["industry"]
    info["Sector in Industry"] = ticker.info["sectorDisp"]
    info["About"] = ticker.info["longBusinessSummary"]
    return str(info)


@tool("officer_info", args_schema=OfficerInfoArgs)
def get_officer_info(stock_name: str) -> str:
    """
    Function which gets info about the officers i.e. CEO,CTO and so on
    """
    ticker = yf.Ticker(get_ticker(stock_name))
    res = ticker.info["companyOfficers"]
    return str(res)


stock_business_tools = [
    get_company_news,
    get_contact_information,
    get_business_info,
    get_officer_info,
]
