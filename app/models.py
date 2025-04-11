from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    job_id: Optional[str] = None  # Lien vers un job

class UserVector(BaseModel):
    user_id: str
    vector: List[float]
    created_at: datetime = datetime.now()

class UserResponse(UserBase):
    id: str
    education_level: str
    work_experience: int

class AgeInterval(BaseModel):
    min: int = Field(..., ge=0)
    max: int = Field(..., ge=0)

    @classmethod
    def validate(cls, v):
        if v["min"] > v["max"]:
            raise ValueError("min age cannot be greater than max age")
        return v

# For JOB Criteria...
class JobBase(BaseModel):
    title: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    place_job: Optional[str] = None
    work_experience: Optional[int] = Field(None, ge=0)
    skills: Optional[List[str]] = []
    age_interval: Optional[AgeInterval] = None
    soft_skills: Optional[List[str]] = []
    education_level: Optional[str] = None
    languages: Optional[List[str]] = []

class JobCreate(JobBase):
    pass

# Pour coonvertir _id en id
class JobOut(BaseModel):
    id: str = Field(..., alias="_id")
    title: str
    description: Optional[str]
    place_job: Optional[str]
    work_experience: Optional[int]
    skills: Optional[List[str]]
    age_interval: Optional[AgeInterval]
    soft_skills: Optional[List[str]]
    education_level: Optional[str]
    languages: Optional[List[str]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class JobUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    place_job: Optional[str] = None
    work_experience: Optional[int] = None
    skills: Optional[List[str]] = None
    age_interval: Optional[AgeInterval]
    soft_skills: Optional[List[str]] = None
    education_level: Optional[str] = None
    languages: Optional[List[str]] = None

class JobDB(JobBase):
    id: str
    created_at: datetime
    updated_at: datetime
    users: List[str] = []  # Liste des IDs utilisateurs associés

