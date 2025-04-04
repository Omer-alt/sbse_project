from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CandidateCreate(BaseModel):
    education: str
    experience: int
    skills: List[str]
    certifications: List[str]
    languages: List[str]
    availability: int
    age: int
    test_score: float

class JobCriteriaWeights(BaseModel):
    education: float
    experience: float
    skills: float
    certifications: float
    language: float
    availability: float
    age: float

class GeneticConfig(BaseModel):
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    X_max: int = 20
    a_min: int = 25
    a_max: int = 50
    
    
class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str

# class UserCreate(UserBase):
#     education_level: str
#     work_experience: int
#     skills: List[str]
#     certifications: List[str]
#     languages: List[str]
#     availability: int
#     age: int
#     test_score: float
#     reference_quality: float
#     soft_skills: float
#     cultural_fit: float
    
class UserCreate(UserBase):
    education_level: str
    work_experience: int
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    availability: int
    age: int
    test_score: float
    reference_quality: float
    soft_skills: float
    cultural_fit: float

class UserVector(BaseModel):
    user_id: str
    vector: List[float]
    created_at: datetime = datetime.now()

class UserResponse(UserBase):
    id: str
    education_level: str
    work_experience: int
