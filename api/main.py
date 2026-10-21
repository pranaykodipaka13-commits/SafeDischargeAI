
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from summarizer import textrank_summarize
from verifier import verify_summary

app = FastAPI(
    title="SafeDischargeAI API",
    description="Clinical text summarization and verification research prototype.",
    version="1.0.0",
)

# Allow the local React development server to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    clinical_note: str = Field(
        min_length=1,
        max_length=100000,
        description="Clinical text to summarize.",
    )
    num_sentences: int = Field(default=8, ge=1, le=30)


@app.get("/")
def root():
    return {
        "message": "SafeDischargeAI API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    note = request.clinical_note.strip()

    if not note:
        raise HTTPException(
            status_code=422,
            detail="Please provide a non-empty clinical note.",
        )

    try:
        summary = textrank_summarize(
            note,
            num_sentences=request.num_sentences,
        )

        verification_df = verify_summary(note, summary)

        verification_records = verification_df.to_dict(orient="records")

        possible_omissions = [
             item
            for item in verification_records
            if item.get("Verification") == "POSSIBLY MISSING"        ]

        return {
            "summary": summary,
            "verification": verification_records,
            "possible_omissions": possible_omissions,
            "verification_count": len(verification_records),
            "possible_omission_count": len(possible_omissions),
            "disclaimer": (
                "Research prototype only. Results require human review "
                "and must not be used as the sole basis for clinical decisions."
            ),
        }

    except Exception as exc:
        # Keep internal exception details out of the API response.
        raise HTTPException(
            status_code=500,
            detail="Summarization failed. Check the backend logs.",
        ) from exc
