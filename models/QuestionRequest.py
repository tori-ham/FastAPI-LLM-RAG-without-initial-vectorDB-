from pydantic import BaseModel
# from pydantic_settings import BaseSettings

class QuestionRequest(BaseModel):
    question : str
    userId : str