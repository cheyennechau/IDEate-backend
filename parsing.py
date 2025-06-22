import ast
from urllib.parse import urlparse
from fastapi import HTTPException

def parse_github_url(url: str) -> tuple[str, str]:
    """
    Given a GitHub repo URL, return (owner, repo). Raises HTTPException(400) if invalid.
    """
    parsed = urlparse(url)
    # Accept github.com or www.github.com
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise HTTPException(status_code=400, detail="Only GitHub URLs are supported")
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid GitHub repo URL")
    owner, repo = parts[0], parts[1]
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        raise HTTPException(status_code=400, detail="Invalid GitHub repo URL")
    return owner, repo

def extract_code_structure_summary(code: str) -> str:
    """
    Parse Python code via ast and extract a brief summary:
    which functions and classes are defined, and top-level imports.
    Returns a short sentence or empty string on parse errors.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return ""
    func_names = []
    class_names = []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    parts = []
    if imports:
        top_mods = {imp.split(".")[0] for imp in imports}
        parts.append(f"Imports: {', '.join(sorted(top_mods))}.")
    if class_names:
        parts.append(f"Defines classes: {', '.join(class_names)}.")
    if func_names:
        parts.append(f"Defines functions: {', '.join(func_names)}.")
    return " ".join(parts)
