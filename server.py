import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
from anthropic import HUMAN_PROMPT, AI_PROMPT
from parsing import parse_github_url, extract_code_structure_summary
import logging
import time

logging.basicConfig(level=logging.INFO)
# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Remove trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Please set ANTHROPIC_API_KEY in your environment")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_URL = "https://api.anthropic.com/v1/messages"

AGENTS = {
    "security": ("security auditor", "review this code for vulnerabilities"),
    "ux": ("UX designer", "critique readability and maintainability of this code"),
    "performance": ("performance engineer", "suggest optimizations for this code"),
    "test": ("test engineer", "identify areas lacking test coverage"),
    "ethics": ("AI ethicist", "evaluate the code for potential bias or ethical concerns"),
    "architecture": ("software architect", "analyze the overall structure and design patterns"),
    "documentation": ("documentation expert", "assess the code documentation and suggest improvements")
}

HEADERS_ANTHROPIC = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

GITHUB_API_HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    GITHUB_API_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

async def call_claude(prompt: str) -> str:
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 800,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Sending to Claude:", payload)
        start = time.time()
        r = await client.post(API_URL, headers=HEADERS_ANTHROPIC, json=payload)
        print("Claude responded in", time.time() - start, "seconds")
        r.raise_for_status()

        try:
            result = r.json()
            print("Claude Response:", result)
            return result["content"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise HTTPException(status_code=500, detail="Error parsing Claude's response")

async def persona_review(role: str, instruction: str, code: str, code_summary: str | None = None) -> str:
    if code_summary:
        prompt = (
            f"As a {role}, {instruction}. "
            f"Here is a brief summary of the code:\n{code_summary}\n\n"
            f"Here is the code:\n{code}"
        )
    else:
        prompt = f"As a {role}, {instruction}:\n\n{code}"
    return await call_claude(prompt)

async def get_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=GITHUB_API_HEADERS)
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Repo not found")
    resp.raise_for_status()
    return resp.json().get("default_branch") or "main"

async def list_python_files(owner: str, repo: str, branch: str, max_size_kb: int = 100) -> list[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=GITHUB_API_HEADERS)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])
    return [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and
           item.get("path", "").endswith(".py") and
           (item.get("size", 0) < max_size_kb * 1024)
    ]

async def fetch_file_content(owner: str, repo: str, branch: str, path: str) -> str:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    print(f"\n🔍 Fetching file from: {raw_url}\n")

    async with httpx.AsyncClient() as client:
        resp = await client.get(raw_url)
    
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"File not found: {raw_url}")
    
    resp.raise_for_status()
    return resp.text

class RepoRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/octocat/Hello-World")

class FileListResponse(BaseModel):
    files: list[str] = Field(..., example=["main.py", "utils/helpers.py"])

class ReviewRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/octocat/Hello-World")
    file_path: str = Field(..., example="main.py")
    branch: str | None = Field(None, example="main")

class ReviewResponse(BaseModel):
    security: str
    ux: str
    performance: str
    test: str
    ethics: str
    architecture: str
    documentation: str
    summary: str

class DebateRequest(BaseModel):
    code: str

class SummaryResponse(BaseModel):
    summary_bullets: str
    
@app.post("/api/repos", response_model=FileListResponse)
async def list_repo_files(req: RepoRequest) -> FileListResponse:
    owner, repo = parse_github_url(req.repo_url)
    branch = await get_default_branch(owner, repo)
    files = await list_python_files(owner, repo, branch)
    return FileListResponse(files=files)

@app.post("/api/review", response_model=ReviewResponse)
async def review_file(req: ReviewRequest) -> ReviewResponse:
    owner, repo = parse_github_url(req.repo_url)
    branch = req.branch or await get_default_branch(owner, repo)
    code = await fetch_file_content(owner, repo, branch, req.file_path)
    code_summary = extract_code_structure_summary(code)

    security_res, ux_res, perf_res, test_res, ethics_res, arch_res, doc_res = await asyncio.gather(
        persona_review("security auditor", "review this code for vulnerabilities", code, code_summary),
        persona_review("UX designer", "critique readability and maintainability of this code", code, code_summary),
        persona_review("performance engineer", "suggest optimizations for this code", code, code_summary),
        persona_review("test engineer", "identify areas lacking test coverage", code, code_summary),
        persona_review("AI ethicist", "evaluate the code for potential bias or ethical concerns", code, code_summary),
        persona_review("software architect", "analyze the overall structure and design patterns", code, code_summary),
        persona_review("documentation expert", "assess the code documentation and suggest improvements", code, code_summary)
    )

    # FIX: Create the action plan using the actual review results
    action_plan = await call_claude(
        f"Here are expert reviews of code:\n\n"
        f"-- Security Review:\n{security_res}\n\n"
        f"-- UX Review:\n{ux_res}\n\n"
        f"-- Performance Review:\n{perf_res}\n\n"
        f"-- Test Review:\n{test_res}\n\n"
        f"-- Ethics Review:\n{ethics_res}\n\n"
        f"-- Architecture Review:\n{arch_res}\n\n"
        f"-- Documentation Review:\n{doc_res}\n\n"
        f"Please provide a concise action plan summarizing the key improvements needed."
    )

    return ReviewResponse(
        security=security_res,
        ux=ux_res,
        performance=perf_res,
        test=test_res,
        ethics=ethics_res,
        architecture=arch_res,
        documentation=doc_res,
        summary=action_plan
    )

@app.post("/api/debate", response_model=dict)
async def debate_code(req: DebateRequest):
    code = req.code
    code_summary = extract_code_structure_summary(code)

    selected_agents = ["security", "ux", "performance", "test", "architecture", "documentation"]
    tasks = [
        persona_review(role, instruction, code, code_summary)
        for key, (role, instruction) in AGENTS.items() if key in selected_agents
    ]
    results = await asyncio.gather(*tasks)

    # map back to keys
    agent_reviews = dict(zip(selected_agents, results))

    action_prompt = "\n\n".join(f"-- {key.capitalize()} Review:\n{text}" for key, text in agent_reviews.items())
    action_plan = await call_claude(f"Here are expert reviews:\n\n{action_prompt}")

    return {
        "structure_summary": code_summary,
        **{f"{key}_review": text for key, text in agent_reviews.items()},
        "action_plan": action_plan,
    }

@app.post("/api/summary",  response_model=SummaryResponse)
async def summarize_reviews(req: ReviewRequest) -> SummaryResponse:
    owner, repo = parse_github_url(req.repo_url)
    branch = req.branch or await get_default_branch(owner, repo)
    code = await fetch_file_content(owner, repo, branch, req.file_path)
    code_summary = extract_code_structure_summary(code)

    selected_agents = ["security", "ux", "performance", "test", "architecture", "documentation"]
    reviews = await asyncio.gather(*[
        persona_review(role, instruction, code, code_summary)
        for key, (role, instruction) in AGENTS.items() if key in selected_agents
    ])

    summary = await call_claude(
        f"Summarize the following reviews into short bullet points:\n\n" +
        "\n\n".join(reviews)
    )
    return SummaryResponse(summary_bullets=summary)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
