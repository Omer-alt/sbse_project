from fastapi import FastAPI
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel
from scipy import stats

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI avec data science!"}

# Exemple avec Pydantic et NumPy
class InputData(BaseModel):
    values: list[float]

@app.post("/mean")
def calculate_mean(data: InputData):
    mean_val = np.mean(data.values)
    return {"mean": mean_val}
