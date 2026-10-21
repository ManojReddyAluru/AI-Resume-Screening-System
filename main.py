import boto3
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from io import BytesIO
from pathlib import Path
from uuid import uuid4
AWS_PROFILE = "resume-screening-dev"
AWS_REGION = "ap-south-1"
S3_BUCKET = "resume-screening-768801480012-2026"

aws_session = boto3.Session(
    profile_name=AWS_PROFILE,
    region_name=AWS_REGION
)

s3_client = aws_session.client("s3")
textract_client = aws_session.client("textract")
dynamodb_table = aws_session.resource("dynamodb").Table("ResumeScreeningResults")
def upload_resume_to_s3(file: UploadFile, content: bytes):
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=file.filename,
        Body=content,
        ContentType="application/pdf"
    )

    return {
        "bucket": S3_BUCKET,
        "key": file.filename,
        "message": "Resume uploaded to S3 successfully"
    }

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobMatchRequest(BaseModel):
    resume_text: str
    job_description: str
    filename: str = "unknown-resume.pdf"

import json
import uuid
from datetime import datetime, timezone


def save_screening_result(filename: str, report: dict):
    candidate_id = str(uuid.uuid4())

    item = {
        "candidate_id": candidate_id,
        "filename": filename,
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "report_json": json.dumps(report)
    }

    dynamodb_table.put_item(Item=item)

    return candidate_id

app = FastAPI(title="AI Resume Screening System")

templates = Jinja2Templates(directory="templates")
@app.get("/")
def home():
    return {
        "message": "AI Resume Screening API is running"
    }





@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF resume."
        )

    try:
        MAX_FILE_SIZE = 5 * 1024 * 1024
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF file must be 5 MB or smaller."
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file."
            )

        reader = PdfReader(BytesIO(content))

        safe_filename = Path(file.filename).name
        s3_key = f"resumes/{uuid4().hex}_{safe_filename}"

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType="application/pdf"
        )

        extraction_method = "Textract"

        try:
            response = textract_client.detect_document_text(
                Document={"Bytes": content}
            )

            text = "\n".join(
                block["Text"]
                for block in response["Blocks"]
                if block["BlockType"] == "LINE"
                and "Text" in block
            )

        except Exception as exc:
            print(f"Textract failed; using local extraction: {exc}")
            extraction_method = "Local PDF extraction"

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text found in the PDF."
            )

        return {
            "filename": safe_filename,
            "pages": len(reader.pages),
            "resume_text": text,
            "s3_bucket": S3_BUCKET,
            "s3_key": s3_key,
            "storage_message": "Resume uploaded to private S3 storage.",
            "text_extraction": extraction_method
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Resume processing failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to process or store this PDF. Check the server terminal."
        )

import re

@app.post("/match-job")
def match_job(data: JobMatchRequest):
    resume = data.resume_text.lower()
    job = data.job_description.lower()

    skills = [
        "python", "java", "c++", "c#", "javascript", "typescript",
        "sql", "mysql", "postgresql", "mongodb", "aws", "azure",
        "google cloud", "machine learning", "deep learning",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "docker", "kubernetes", "git", "github", "react",
        "fastapi", "flask", "django", "html", "css",
        "rest api", "rest apis", "api development",
        "data structures", "algorithms", "communication",
        "problem solving", "data analysis", "linux",
        "computer networks", "database management"
    ]

    def contains_skill(text, skill):
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        return re.search(pattern, text) is not None

    required = [
        skill for skill in skills
        if contains_skill(job, skill)
    ]

    matched = [
        skill for skill in required
        if contains_skill(resume, skill)
    ]

    missing = [
        skill for skill in required
        if skill not in matched
    ]

    score = (
        round(len(matched) / len(required) * 100, 2)
        if required else 0
    )

    similarity = 0

    if resume.strip() and job.strip():
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume, job])

        similarity = round(
            cosine_similarity(
                vectors[0:1], vectors[1:2]
            )[0][0] * 100,
            2
        )

    combined_score = round(
        (0.7 * score) + (0.3 * similarity),
        2
    )

    return {
        "required_skills": required,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_percentage": score,
        "text_similarity_percentage": similarity,
        "combined_match_percentage": combined_score
    }
@app.post("/download-report")
def download_report(data: dict):
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)

    pdf.setTitle("Resume Screening Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 790, "Resume Screening Report")

    y = 750
    fields = [
        ("Candidate", data.get("filename", "Resume")),
        ("Skill Match", f'{data.get("skill_match_percentage", 0)}%'),
        ("Text Similarity", f'{data.get("text_similarity_percentage", 0)}%'),
        ("Overall Match", f'{data.get("combined_match_percentage", 0)}%'),
        ("Matched Skills", ", ".join(data.get("matched_skills", [])) or "None"),
        ("Missing Skills", ", ".join(data.get("missing_skills", [])) or "None"),
    ]

    for label, value in fields:
        text = f"{label}: {value}"
        for i in range(0, len(text), 85):
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y = 790
            pdf.setFont("Helvetica", 11)
            pdf.drawString(50, y, text[i:i + 85])
            y -= 20

    pdf.save()
    pdf_buffer.seek(0)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="screening_report.pdf"'
        }
    )
@app.post("/verify-skills")

def verify_skills(data: JobMatchRequest):
    import re

    resume = data.resume_text.lower()
    job = data.job_description.lower()

    skills = [
        "python", "java", "c++", "c#", "javascript", "typescript",
        "sql", "mysql", "postgresql", "mongodb", "aws", "azure",
        "google cloud", "machine learning", "deep learning",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "docker", "kubernetes", "git", "github", "react",
        "fastapi", "flask", "django", "html", "css",
        "rest api", "rest apis", "api development",
        "data structures", "algorithms", "communication",
        "problem solving", "data analysis", "linux",
        "computer networks", "database management"
    ]

    action_words = [
        "developed", "built", "implemented", "created", "trained",
        "deployed", "designed", "used", "programmed", "integrated",
        "tested", "analyzed", "analysed", "wrote", "configured",
        "managed", "maintained", "automated", "optimized",
        "optimised", "processed", "predicted", "evaluated"
    ]

    evidence_sections = [
        "projects", "project experience", "academic projects",
        "personal projects", "experience", "work experience",
        "professional experience", "internship", "education"
    ]

    def contains_skill(text, skill):
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        return re.search(pattern, text) is not None

    # Identify skills relevant to the job and present in the resume.
    required_skills = [
        skill for skill in skills
        if contains_skill(job, skill)
    ]

    lines = [
        line.strip()
        for line in resume.splitlines()
        if line.strip()
    ]

    results = []

    for skill in required_skills:
        skill_pattern = (
            r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        )

        mentions = [
            line for line in lines
            if re.search(skill_pattern, line)
        ]

        evidence_sentences = []

        for line in mentions:
            if any(
                re.search(
                    r"\b" + re.escape(action) + r"\b",
                    line
                )
                for action in action_words
            ):
                evidence_sentences.append(line)

        # Also inspect adjacent lines for project descriptions
        # where the skill and action appear on separate lines.
        for index, line in enumerate(lines):
            if not re.search(skill_pattern, line):
                continue

            start = max(0, index - 1)
            end = min(len(lines), index + 2)

            context = " ".join(lines[start:end])

            if (
                any(
                    re.search(
                        r"\b" + re.escape(action) + r"\b",
                        context
                    )
                    for action in action_words
                )
                and context not in evidence_sentences
            ):
                evidence_sentences.append(context)

        if evidence_sentences:
            results.append({
                "skill": skill,
                "status": "Evidence sentence found",
                "evidence_context": " ".join(
                    evidence_sentences[:2]
                )
            })

        elif mentions:
            results.append({
                "skill": skill,
                "status": "Mentioned only; evidence unclear",
                "evidence_context": " ".join(mentions[:2])
            })

        else:
            results.append({
                "skill": skill,
                "status": "Not found in resume",
                "evidence_context": ""
            })

    return {
        "skill_analysis": results,
        "note": (
            "Evidence is identified using text-matching rules. "
            "A matching sentence does not independently verify "
            "actual proficiency or the truth of a claim."
        )
    }

@app.post("/candidate-report")
def candidate_report(data: JobMatchRequest):
    match_result = match_job(data)
    evidence_result = verify_skills(data)

    report = {
        "report_type": "AI Resume Screening Report",
        "candidate_evaluation": {
            "skill_match_percentage": match_result[
                "skill_match_percentage"
            ],
            "text_similarity_percentage": match_result[
                "text_similarity_percentage"
            ],
            "combined_match_percentage": match_result[
                "combined_match_percentage"
            ],
            "matched_skills": match_result["matched_skills"],
            "missing_skills": match_result["missing_skills"],
            "required_skills": match_result["required_skills"]
        },
        "skill_matching": match_result,
        "evidence_analysis": evidence_result,
        "disclaimer": (
            "This is a preliminary automated assessment. "
            "The combined score is an experimental indicator, "
            "not a validated prediction of job performance. "
            "Verify evidence and review candidates fairly "
            "before making hiring decisions."
        )
    }

    try:
        candidate_id = save_screening_result(
            filename=data.filename,
            report=report
        )
        report["candidate_id"] = candidate_id
        report["storage_status"] = "Saved to DynamoDB"

    except Exception as exc:
        print(f"DynamoDB save failed: {exc}")
        report["storage_status"] = "Not saved to DynamoDB"

    return report


@app.get("/screen")
def screening_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/candidate-summary")
def candidate_summary(data: JobMatchRequest):
    match_result = match_job(data)
    evidence_result = verify_skills(data)

    matched = match_result["matched_skills"]
    missing = match_result["missing_skills"]
    score = match_result["skill_match_percentage"]

    if score >= 80:
        recommendation = "Strong skills alignment"
    elif score >= 50:
        recommendation = "Moderate skills alignment"
    else:
        recommendation = "Limited skills alignment"

    return {
        "candidate_summary": {
            "overall_assessment": recommendation,
            "skill_match_percentage": score,
            "strengths": matched,
            "skill_gaps": missing,
            "evidence_review": evidence_result["skill_analysis"],
            "suggested_interview_questions": [
                f"Can you describe a project where you used {skill}?"
                for skill in missing[:3]
            ]
        },
        "disclaimer": (
            "This is a preliminary, rule-based assessment, "
            "not an AI-verified judgment of candidate ability. "
            "Review the resume and assess candidates fairly."
        )
    }

@app.post("/compare-resumes")
async def compare_resumes(
    files: list[UploadFile] = File(...),
    job_description: str = Form(...)
):
    from io import BytesIO

    candidates = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        try:
            pdf = PdfReader(BytesIO(await file.read()))
            resume_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()

            if not resume_text:
                candidates.append({
                    "filename": file.filename,
                    "error": "No extractable text found in this PDF."
                })
                continue

            request_data = JobMatchRequest(
                resume_text=resume_text,
                job_description=job_description
            )

            match_result = match_job(request_data)
            evidence_result = verify_skills(request_data)

            candidates.append({
                "filename": file.filename,
                "skill_match_percentage":
                    match_result["skill_match_percentage"],
                "text_similarity_percentage":
                    match_result["text_similarity_percentage"],
                "matched_skills": match_result["matched_skills"],
                "missing_skills": match_result["missing_skills"],
                "evidence_analysis":
                    evidence_result["skill_analysis"]
            })

        except Exception:
            candidates.append({
                "filename": file.filename,
                "error": "Unable to process this PDF."
            })

    successful_candidates = [
        candidate for candidate in candidates if "error" not in candidate
    ]

    successful_candidates.sort(
        key=lambda candidate: candidate["skill_match_percentage"],
        reverse=True
    )

    return {
        "job_description": job_description,
        "candidate_count": len(successful_candidates),
        "ranking_basis": "Skill match percentage, descending",
        "candidates": successful_candidates,
        "errors": [
            candidate for candidate in candidates if "error" in candidate
        ],
        "disclaimer": (
            "Rankings are preliminary rule-based comparisons, "
            "not hiring decisions. Review candidates fairly."
        )
    }
@app.get("/screening-history")
def screening_history():
    try:
        response = dynamodb_table.scan(
            ProjectionExpression="#cid, filename, screened_at",
            ExpressionAttributeNames={
                "#cid": "candidate_id"
            }
        )

        items = response.get("Items", [])

        while "LastEvaluatedKey" in response:
            response = dynamodb_table.scan(
                ProjectionExpression="#cid, filename, screened_at",
                ExpressionAttributeNames={
                    "#cid": "candidate_id"
                },
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items.extend(response.get("Items", []))

        items.sort(
            key=lambda item: item.get("screened_at", ""),
            reverse=True
        )

        return {
            "total_screenings": len(items),
            "screenings": items
        }

    except Exception as exc:
        print(f"Screening history retrieval failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve screening history."
        )
    

@app.get("/github-profile/{username}")
def github_profile(username: str):
    import re
    import urllib.request
    import urllib.error

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", username):
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub username."
        )

    url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10&type=owner"

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ResumeScreeningProject"
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            repositories = json.loads(response.read().decode("utf-8"))

        return {
            "username": username,
            "repository_count": len(repositories),
            "repositories": [
                {
                    "name": repo["name"],
                    "description": repo.get("description"),
                    "language": repo.get("language"),
                    "url": repo["html_url"],
                    "stars": repo.get("stargazers_count", 0)
                }
                for repo in repositories
            ]
        }

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(
                status_code=404,
                detail="GitHub user not found."
            )
        if exc.code in (403, 429):
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit reached. Try again later."
            )
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API returned HTTP {exc.code}."
        )

    except Exception as exc:
        print(f"GitHub API request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API request failed: {type(exc).__name__}: {exc}"
        )
