from fastapi import (
    FastAPI,
)
import sys

sys.path.append("fast_api")

from app.api.genai import support

app = FastAPI(
    title="Support Ticket Processing API",
    description="API for processing support tickets using LangGraph workflow",
    version="1.0.0",  # Version as string
)


app.include_router(
    support,
    prefix="/v1/support",
    tags=["support"],
)
