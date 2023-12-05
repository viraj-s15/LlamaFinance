from pydantic.v1 import BaseModel, Field


class GetTickerArgs(BaseModel):
    company_name: str = Field(
        description="Should be the name of the company whose stock you want to get the ticker for"
    )


class CompanyNewsArgs(BaseModel):
    stock_name: str = Field(
        description="Should be the name of a company whose news you want to get"
    )


class GenericStockInfoArgs(BaseModel):
    stock_name: str = Field(
        description="Should be the name of a company whose stock information you want to get"
    )
