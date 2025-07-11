from typing import Optional
from pydantic import BaseModel


class RawItem(BaseModel):
    user : Optional[str] = None
    card : Optional[str] = None
    year : Optional[str] = None
    month : Optional[str] = None
    day : Optional[str] = None
    time : Optional[str] = None
    amount: Optional[str] = None
    use_chip: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    zip : Optional[str] = None
    mcc : Optional[str] = None
    errors : Optional[str] = None

    class Config:
        extra = "ignore"
