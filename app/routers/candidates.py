from fastapi import APIRouter, HTTPException
from bson import ObjectId
from app.models import CandidateCreate
from app.database import users_collection, vectors_collection, jobs_collection
from app.models import UserCreate, UserResponse, UserVector

router = APIRouter(tags=["Candidates"])

# Base de données simulée
candidates_db = []
job_criteria = None

education_mapping = {
    "Level 1": 1,
    "Level 2": 2,  
    "Level 3": 3,
    "Bachelor": 3,
    "Level 4": 4,
    "Level 5": 5,
    "Master": 5,
    "Doctor first years": 6,
    "Doctor second years": 7,
    "Doctor third years": 8,
    "Doctor": 8
}

# @router.post("/candidates/")
# async def create_candidate(candidate: CandidateCreate):
#     candidates_db.append(candidate.dict())
#     return {"message": "Candidate added successfully"}

# @router.get("/candidates/")
# async def get_all_candidates():
#     return candidates_db


router = APIRouter(tags=["Users"])    
    
@router.post("/users/")
async def create_user(user: UserCreate):
    try:
        if user.education_level not in education_mapping:
            raise HTTPException(
                400,
                f"Niveau d'éducation '{user.education_level}' non reconnu. Valeurs acceptées: {list(education_mapping.keys())}"
            )

        vector = [
            education_mapping[user.education_level],
            user.work_experience,
            len(user.skills or []),
            user.test_score,
            len(user.certifications or []),
            len(user.languages or []),
            user.availability,
            user.age,
            user.reference_quality,
            user.soft_skills,
            user.cultural_fit
        ]
        
        print("Vecteur utilisateur :", vector)
        user_dict = user.dict()
        result = await users_collection.insert_one(user_dict)
        print("Un utilisateur... :", str(result.inserted_id))
        
        vector_doc = UserVector(
            user_id=str(result.inserted_id),
            vector=vector
        )
        await vectors_collection.insert_one(vector_doc.dict())
        
        # Supprime le champ _id du dict d'origine (non sérialisable)
        user_dict.pop("_id", None)

        return {**user_dict, "id": str(result.inserted_id)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur lors de la création de l'utilisateur: {str(e)}")



@router.get("/users/")  # UserOut est le modèle de sortie
async def list_users():
    users = []
    cursor = users_collection.find({})
    async for user in cursor:
        user["id"] = str(user["_id"])
        del user["_id"]
        users.append(user)
    return users

@router.get("/users/vectors/")  # UserOut est le modèle de sortie
async def list_users():
    users = []
    cursor = vectors_collection.find({})
    async for user in cursor:
        user["id"] = str(user["_id"])
        del user["_id"]
        users.append(user)
    return users


@router.put("/users/{user_id}/assign-job")
async def assign_user_to_job(user_id: str, job_id: str):
    try:
        # Vérifier que le job existe
        job = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(404, "Job not found")
        
        # Mettre à jour l'utilisateur
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"job_id": job_id}}
        )
        
        # Ajouter l'utilisateur au job
        await jobs_collection.update_one(
            {"_id": ObjectId(job_id)},
            {"$addToSet": {"users": user_id}}
        )
        
        return {"message": "User successfully assigned to job"}
    except Exception as e:
        raise HTTPException(400, f"Assignment failed: {str(e)}")
