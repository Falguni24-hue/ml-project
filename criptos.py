from pydantic import BaseModel, Field,ConfigDict

class Cripto(BaseModel):
    h24: float = Field(alias="24h")
    d7: float = Field(alias="7d")
    h24_volume: float = Field(alias="24h_volume")
    mkt_cap: float

    class Config:
        validate_by_name = True
