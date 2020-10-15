# http://127.0.0.1:8000/docs
# uvicorn main:app --reload

from typing import Optional
from fastapi import FastAPI
import immothep_fct as fct


app = FastAPI()

@app.get("/")
def read_root():
    return {"welcome_message": "Bienvenue sur Immothep !"}


@app.get("/estimation/{cp}/{surface}/{terrain}/{nbpieces}/")
def api_estimate(cp: int, surface: int, terrain: int, nbpieces: int):
    estimation = fct.estimation(cp, surface, terrain, nbpieces)
    return {'estimation': estimation}

