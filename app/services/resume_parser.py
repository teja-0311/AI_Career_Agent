"""Resume Parser Service with explicit backend modes: 'gemini' and 'rule_based'.

In 'gemini' mode, never silently degrades to rule-based fallback on failure.
"""

import io
import re
from typing import Optional
from pypdf import PdfReader

try:
    import docx
except ImportError:
    docx = None

from app.config import settings
from app.services.llm import GeminiClient, get_gemini_client
from app.schemas.resume import CandidateProfile


class ResumeParserService:
    def __init__(self, backend: Optional[str] = None, llm_client: Optional[GeminiClient] = None):
        self.backend = (backend or settings.resume_parser_backend).lower()
        if self.backend not in ["gemini", "rule_based"]:
            raise ValueError(f"Unsupported RESUME_PARSER_BACKEND: '{self.backend}'. Must be 'gemini' or 'rule_based'.")
        self.llm_client = llm_client or get_gemini_client()

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts).strip()
        except Exception as e:
            raise ValueError(f"Failed to read PDF document: {e}")

    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        if docx is None:
            raise RuntimeError("python-docx is not installed.")
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as e:
            raise ValueError(f"Failed to read DOCX document: {e}")

    def extract_raw_text(self, filename: str, file_bytes: bytes) -> str:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise ValueError("Uploaded resume file is empty.")

        fname_lower = filename.lower()
        if fname_lower.endswith(".pdf"):
            text = self.extract_text_from_pdf(file_bytes)
        elif fname_lower.endswith(".docx"):
            text = self.extract_text_from_docx(file_bytes)
        elif fname_lower.endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="replace").strip()
        else:
            raise ValueError("Unsupported resume format. Please upload PDF, DOCX, or TXT.")

        if not text:
            raise ValueError("No readable text could be extracted from the uploaded document.")
        return text

    def rule_based_extract(self, raw_text: str) -> CandidateProfile:
        """Deterministic extractor providing structured profile fallback without hallucination."""
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
        email = email_match.group(0) if email_match else None

        phone_match = re.search(r"\(?\+?\d{1,3}\)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
        phone = phone_match.group(0) if phone_match else None

        prog_langs = ["Python", "TypeScript", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "HTML", "CSS"]
        frameworks = ["FastAPI", "React", "Next.js", "Django", "Flask", "Node.js", "LangChain", "LangGraph", "PyTorch", "TensorFlow", "Docker", "Kubernetes"]
        databases = ["PostgreSQL", "ChromaDB", "MongoDB", "Redis", "MySQL", "SQLite"]

        raw_lower = raw_text.lower()
        found_langs = [p for p in prog_langs if re.search(r"\b" + re.escape(p.lower()) + r"\b", raw_lower)]
        found_fw = [f for f in frameworks if re.search(r"\b" + re.escape(f.lower()) + r"\b", raw_lower)]
        found_db = [d for d in databases if re.search(r"\b" + re.escape(d.lower()) + r"\b", raw_lower)]
        all_skills = list(dict.fromkeys(found_langs + found_fw + found_db))

        level = None
        years = None
        year_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s*of\s*experience)?", raw_lower)
        if year_matches:
            try:
                years = max(float(y) for y in year_matches)
            except ValueError:
                pass

        if years is not None:
            if years >= 5.0:
                level = "Senior"
            elif years >= 2.0:
                level = "Mid-level"
            else:
                level = "Entry-level"
        else:
            if "senior" in raw_lower:
                level = "Senior"
            elif "lead" in raw_lower:
                level = "Lead"
            elif "intern" in raw_lower or "entry level" in raw_lower or "junior" in raw_lower:
                level = "Entry-level"

        work_mode = None
        if "remote" in raw_lower:
            work_mode = "Remote"
        elif "hybrid" in raw_lower:
            work_mode = "Hybrid"
        elif "on-site" in raw_lower or "onsite" in raw_lower:
            work_mode = "On-site"

        return CandidateProfile(
            email=email,
            phone=phone,
            programming_languages=found_langs,
            frameworks=found_fw,
            databases=found_db,
            technical_skills=all_skills,
            skills=all_skills,
            technologies=all_skills,
            experience_level=level,
            years_of_experience=years,
            preferred_work_mode=work_mode,
            work_mode_preference=work_mode,
            raw_text=raw_text,
        )

    def parse(self, raw_text: str) -> CandidateProfile:
        """Parse resume into CandidateProfile.

        In 'gemini' mode, raises clear errors on missing credentials, API failures, or schema errors.
        In 'rule_based' mode, runs deterministic extraction.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Resume text is empty.")

        if self.backend == "gemini":
            if not self.llm_client or not self.llm_client.is_available():
                raise RuntimeError(
                    "RESUME_PARSER_BACKEND is set to 'gemini', but GEMINI_API_KEY is not set or invalid. "
                    "Set GEMINI_API_KEY in .env or configure RESUME_PARSER_BACKEND=rule_based."
                )

            prompt = (
                "Extract structured candidate profile details from the resume below into the requested JSON schema.\n"
                "Strict Guidelines:\n"
                "- Extract skills, degrees, universities, projects, and work history ONLY if explicitly present in the text.\n"
                "- Never fabricate companies, qualifications, or experience not stated.\n"
                "- Categorize technical skills into programming languages, frameworks, and databases.\n"
                "- Experience levels allowed: Entry-level, Mid-level, Senior, Lead. If unknown, leave null.\n\n"
                f"Resume Content:\n{raw_text}"
            )
            try:
                profile = self.llm_client.generate_structured(prompt=prompt, schema=CandidateProfile)
                profile.raw_text = raw_text
                return profile
            except Exception as e:
                raise RuntimeError(f"Gemini resume parsing failed: {str(e)}")

        return self.rule_based_extract(raw_text)
