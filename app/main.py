import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": f"Hello, FastAPI in Docker! Host: {os.getenv('HOST', '0.0.0.0')} and Port: {os.getenv('PORT', '8000')}"}
