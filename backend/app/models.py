"""
this module is dedicated for the abstract classes and database models of the application
"""
from pydantic import BaseModel, ConfigDict,Field
from typing import Optional,  List,  Any
from datetime import datetime
from typing_extensions import Annotated
from pydantic.functional_validators  import BeforeValidator


PyObjectId = Annotated[str, BeforeValidator(str)]


class PeopleModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id",default=None)
    first_name: str= Field(...)
    last_name: str = Field(...)
    phone_number: str= Field(...)
    birth_date: datetime=Field(...,description="The user's birthdate in YYYY-MM-DD format")
    role: str= Field(...)
    person_id: str= Field(...)
    model_config= ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example":{
                "first_name":"test",
                "last_name":"subject",
                "phone_number":"919199191991",
                "birth_date": datetime.now(),
                "role":"student",
                "person_id":"100"
            }
        },
    )
def person_serializer(person:dict)-> dict:
    if isinstance(person.get('birth_date'), datetime):
        person['birth_date'] = person['birth_date'].strftime("%Y-%m-%d %H:%M:%S")
    
    return{
        "id":str(person["_id"]),
        "first_name":person["first_name"],
        "last_name":person["last_name"],
        "phone_number":person["phone_number"],
        "birth_date":person["birth_date"],
        "role":person["role"],
        "person_id":person["person_id"]
    }



class TrackingModel(BaseModel):
    person_id: Optional[str] = "unprocessed"  # Default to "unprocessed" if not provided
    time_entered: datetime
    time_exited: Optional[datetime] = None
    face_images: Optional[List[str]] = None
    frame_images: Optional[List[str]]=None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "person_id": "unprocessed",
                "time_entered": datetime.now(),
                "time_exited": datetime.now(),
                "face_images": [
                    "/path/to/image1.jpg",
                    "/path/to/image2.jpg"
                ],
                 "frame_images": [
                    "/path/to/image1.jpg",
                    "/path/to/image2.jpg"
                ],
            }
        }
def tracking_serializer(tracking: dict) -> dict:
    # Ensure the datetime fields are not converted to string
    return {
        "person_id": tracking.get("person_id", "unprocessed"),  # Default to "unprocessed" if not provided
        "time_entered": tracking["time_entered"],  # Keep as datetime object
        "time_exited": tracking.get("time_exited",None),  # Keep as datetime object or None
        "face_images": tracking.get("face_images", []),  # Default to empty list if not provided
        "frame_images": tracking.get("frame_images", [])  # Default to empty list if not provided
    }
