from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

users_collection = db.users
vectors_collection = db.user_vectors

# Ajouter la collection jobs
jobs_collection = db.jobs