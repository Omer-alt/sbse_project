# from fastapi import APIRouter, HTTPException
# from app.models import GeneticConfig

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd

from app.services.genetic import GeneticAlgorithm, plot_all_candidates_evolution_with_id

router = APIRouter(tags=["Genetic Algorithm"])

class GeneticRequest(BaseModel):
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    X_max: int = 20
    a_min: int = 25
    a_max: int = 50
    weights: Optional[List[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

@router.post("/optimize/genetic")
async def genetic_optimization(request: GeneticRequest):
    try:
        # Utiliser les poids par défaut si non fournis
        default_weights = [0.1, 0.15, 0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05]
        weights = request.weights if request.weights else default_weights
        
        # Initialiser l'algorithme avec les paramètres fournis
        ga = GeneticAlgorithm(
            population_size=10,  # Taille fixe selon votre exemple
            generations=request.generations,
            mutation_rate=request.mutation_rate,
            crossover_rate=request.crossover_rate,
            X_max=request.X_max,
            a_min=request.a_min,
            a_max=request.a_max
        )

        # Exécuter l'optimisation
        ranked_candidates, final_scores = await ga.run(weights)
        
        # Formater la réponse
        response = {
            "ranked_candidates": [
                {
                    "rank": rank,
                    "id": cid,
                    "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    "score": float(score)
                }
                for rank, (cid, vector, score) in enumerate(ranked_candidates, 1)
            ],
            "final_scores": {k: float(v) for k, v in final_scores.items()}
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Genetic optimization failed: {str(e)}"
        )