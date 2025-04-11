from fastapi import APIRouter, HTTPException, Depends
from typing import List
from bson import ObjectId
from datetime import datetime
from app.database import jobs_collection
from app.models import JobCreate, JobDB, JobUpdate

router = APIRouter(tags=["Jobs Ad"])

# Helper pour la conversion ObjectId
def format_job(job: dict) -> dict:
    job["id"] = str(job["_id"])
    del job["_id"]
    return job

@router.post("/jobs/", response_model=JobDB)
async def create_job(job: JobCreate):
    try:
        job_data = job.dict()
        job_data.update({
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "users": []
        })
        
        result = await jobs_collection.insert_one(job_data)
        new_job = await jobs_collection.find_one({"_id": result.inserted_id})
        return format_job(new_job)
        
    except Exception as e:
        raise HTTPException(500, f"Job creation failed: {str(e)}")

@router.get("/jobs/", response_model=List[JobDB])
async def get_all_jobs(limit: int = 10, skip: int = 0):
    try:
        jobs = await jobs_collection.find().skip(skip).limit(limit).to_list(None)
        return [format_job(job) for job in jobs]
    except Exception as e:
        raise HTTPException(500, f"Error fetching jobs: {str(e)}")

@router.get("/jobs/{job_id}", response_model=JobDB)
async def get_job(job_id: str):
    try:
        job = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(404, "Job not found")
        return format_job(job)
    except Exception as e:
        raise HTTPException(400, f"Invalid job ID: {str(e)}")

@router.patch("/jobs/{job_id}", response_model=JobDB)
async def update_job(job_id: str, job_update: JobUpdate):
    try:
        update_data = {k: v for k, v in job_update.dict().items() if v is not None}
        update_data["updated_at"] = datetime.now()
        
        result = await jobs_collection.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(404, "No changes detected or job not found")
            
        updated_job = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        return format_job(updated_job)
    except Exception as e:
        raise HTTPException(400, f"Update failed: {str(e)}")

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    try:
        # Supprimer le job
        delete_result = await jobs_collection.delete_one({"_id": ObjectId(job_id)})
        
        if delete_result.deleted_count == 0:
            raise HTTPException(404, "Job not found")
            
        # Retirer les références utilisateurs
        await users_collection.update_many(
            {"job_id": job_id},
            {"$set": {"job_id": None}}
        )
        
        return {"message": "Job deleted successfully"}
    except Exception as e:
        raise HTTPException(400, f"Deletion failed: {str(e)}")