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


class ContactInfoArgs(BaseModel):
    stock_name: str = Field(
        description="Should be the name of a company whose contact information you want to get"
    )


class BusinessInfoArgs(BaseModel):
    stock_name: str = Field(
        description="Should be the name of a company whose business information you want to get"
    )


class OfficerInfoArgs(BaseModel):
    stock_name: str = Field(
        description="Should be the name of a company whose officer information you want to get"
    )
