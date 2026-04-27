import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import configure_langsmith, GROQ_API_KEY
from agent import create_farm_agent

configure_langsmith()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="FarmAI", version="1.0.0", description="LangChain-powered agricultural assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


def _get_api_key() -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing GROQ_API_KEY. Set it in .env.")
    return GROQ_API_KEY


def _extract_tools_used(intermediate_steps: list) -> list[str]:
    return [step[0].tool for step in intermediate_steps]


# ---------- Request models ----------

class PestAnalysisRequest(BaseModel):
    crop: str
    symptoms: str


class SchemeRequest(BaseModel):
    state: str
    crop: str
    category: str


# ---------- Endpoints ----------

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/pest-analysis")
async def pest_analysis(body: PestAnalysisRequest):
    api_key = _get_api_key()
    query = (
        f"My {body.crop} crop has these symptoms: {body.symptoms}. "
        "What pest is causing this and how should I treat it?"
    )
    logger.info("pest-analysis | crop=%s", body.crop)

    try:
        executor = create_farm_agent(api_key)
        result = executor.invoke(
            {"input": query},
            config={
                "run_name": "pest-analysis",
                "tags": ["pest-analysis", body.crop],
                "metadata": {
                    "endpoint": "/pest-analysis",
                    "crop": body.crop,
                    "symptoms": body.symptoms,
                },
            },
        )
    except Exception as exc:
        logger.exception("pest-analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "crop": body.crop,
        "symptoms": body.symptoms,
        "analysis": result["output"],
        "tools_invoked": _extract_tools_used(result.get("intermediate_steps", [])),
    }


@app.post("/scheme-recommendation")
async def scheme_recommendation(body: SchemeRequest):
    api_key = _get_api_key()
    query = (
        f"I am a {body.category} farmer in {body.state} growing {body.crop}. "
        "What government schemes and subsidies am I eligible for?"
    )
    logger.info("scheme-recommendation | state=%s crop=%s category=%s", body.state, body.crop, body.category)

    try:
        executor = create_farm_agent(api_key)
        result = executor.invoke(
            {"input": query},
            config={
                "run_name": "scheme-recommendation",
                "tags": ["scheme-recommendation", body.state, body.crop],
                "metadata": {
                    "endpoint": "/scheme-recommendation",
                    "state": body.state,
                    "crop": body.crop,
                    "category": body.category,
                },
            },
        )
    except Exception as exc:
        logger.exception("scheme-recommendation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "state": body.state,
        "crop": body.crop,
        "category": body.category,
        "schemes": result["output"],
        "tools_invoked": _extract_tools_used(result.get("intermediate_steps", [])),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "FarmAI"}
