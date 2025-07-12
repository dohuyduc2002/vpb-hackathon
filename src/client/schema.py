from typing import Optional
from pydantic import BaseModel


class RawItem(BaseModel):
    user : str 
    card : str 
    year : str 
    month : str 
    day : str 
    time : str 
    amount: str 
    use_chip: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    zip : Optional[str] = None
    mcc : str
    errors : Optional[str] = None

    class Config:
        extra = "ignore"


