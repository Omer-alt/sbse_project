from fastapi import FastAPI
from .routers import candidates, evaluation, genetic

app = FastAPI(title="Intelligent Recruitment System")

app.include_router(candidates.router)
app.include_router(evaluation.router)
app.include_router(genetic.router)

@app.get("/")
def health_check():
    return {
        "status": "active",
        "modules": ["AHP", "Genetic Algorithm", "Candidate Management"]
    }