from fastapi import APIRouter, HTTPException
from app.models import JobCriteriaWeights
from app.services import ahp

router = APIRouter(tags=["AHP Evaluation"])

@router.post("/criteria/")
async def set_criteria(weights: JobCriteriaWeights):
    global job_criteria
    job_criteria = weights.dict()
    return {"message": "Criteria updated successfully"}

@router.post("/evaluate/ahp")
async def run_ahp_evaluation():
    if not job_criteria:
        raise HTTPException(400, "Job criteria not set")
    
    weights = list(job_criteria.values())
    A, weights_df, cr, rgmm = ahp.AHP_1_Participant(weights)
    
    return {
        "consistency_ratio": cr,
        "weights": weights_df.to_dict(),
        "comparison_matrix": A.tolist()
    }