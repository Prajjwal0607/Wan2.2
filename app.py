from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    task: str = "t2v-A14B"
    size: str = "1280*720"
    ckpt_dir: str = "./Wan2.2-T2V-A14B"

@app.post("/generate")
async def generate_video(req: GenerationRequest):
    cmd = [
        "python", "generate.py",
        "--task", req.task,
        "--size", req.size,
        "--ckpt_dir", req.ckpt_dir,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--prompt", req.prompt
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"stdout": result.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Generation timed out")
