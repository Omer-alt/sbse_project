from fastapi import FastAPI
from .routers import candidates, evaluation, genetic, jobs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Intelligent Recruitment System")

app.include_router(candidates.router)
app.include_router(evaluation.router)
app.include_router(genetic.router)
app.include_router(jobs.router)

# Configuration du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À utiliser uniquement en développement !
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "status": "active",
        "modules": ["AHP", "Genetic Algorithm", "Candidate Management"]
    }