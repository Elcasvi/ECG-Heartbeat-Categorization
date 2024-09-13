from pydantic import BaseModel
class ECGInput(BaseModel):
    data: list[float]