"""FastAPI route handlers for resume upload and parsing."""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config import settings
from app.schemas.resume import CandidateProfile
from app.services.resume_parser import ResumeParserService

router = APIRouter(prefix="/resume", tags=["Resume"])
parser_service = ResumeParserService()


@router.post(
    "/upload",
    response_model=CandidateProfile,
    summary="Upload and parse a PDF, DOCX, or TXT resume",
    status_code=status.HTTP_200_OK,
)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file (.pdf, .docx, or .txt)"),
):
    """Streams file checking byte limits and safely parses candidate profile."""
    filename = file.filename or "unknown"
    fn_lower = filename.lower()
    if not (fn_lower.endswith(".pdf") or fn_lower.endswith(".docx") or fn_lower.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file.",
        )

    # Enforce upload limits while reading
    chunk_size = 64 * 1024
    content = bytearray()
    total_read = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > settings.upload_max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.upload_max_bytes // (1024 * 1024)} MB.",
            )
        content.extend(chunk)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        raw_text = parser_service.extract_raw_text(filename, bytes(content))
        profile = parser_service.parse(raw_text)
        return profile
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the resume document: {str(e)}",
        )
