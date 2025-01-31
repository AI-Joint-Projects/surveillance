"""
This Module is dedicated for the input sschema of the application endpoints
"""
from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime


class PersonRegister(BaseModel):
    person_id: str
    first_name: str 
    last_name: str 
    role: str
    birth_date: datetime
    phone_number:str

    

class PersonRemove(BaseModel):
    person_id: int

class Surveillance(BaseModel):
    start_time: datetime
    end_time: datetime

   
