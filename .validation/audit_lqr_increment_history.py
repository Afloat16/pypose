import ast
import hashlib
import json
import subprocess

PRODUCTION = "23853bda897abc44180471cc050c5c9f50abd0d9"
PATH = "pypose/module/lqr.py"
revisions = subprocess.check_output(
    ["git", "rev-list", "--all", "--full-history", "--", PATH], text=True
).splitlines()
blobs = {}
for revision in revisions:
    if revision == PRODUCTION:
        continue
    path = revision + ":" + PATH
    result = subprocess.run(["git", "rev-parse", path], text=True, capture_output=True)
    if result.returncode:
        continue
    sha = result.stdout.strip()
    if sha in blobs:
        blobs[sha]["commits"].append(revision)
        continue
    source = subprocess.check_output(["git", "show", path], text=True)
    methods = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LQR":
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in ("forward", "lqr_forward"):
                        methods.append(ast.get_source_segment(source, child))
    except SyntaxError:
        methods = []
    blobs[sha] = {"commits": [revision], "methods": methods}
unique_methods = {}
for sha, record in blobs.items():
    for method in record["methods"]:
        digest = hashlib.sha256(method.encode()).hexdigest()
        item = unique_methods.setdefault(digest, {"source": method, "blobs": []})
        item["blobs"].append(sha)
result = {"path": PATH, "production_excluded": PRODUCTION,
          "revisions": len(revisions), "blobs": len(blobs),
          "unique_methods": len(unique_methods),
          "method_variants": unique_methods, "source_blobs": blobs}
with open("lqr-increment-history.json", "w") as output:
    json.dump(result, output, indent=2)
print("LQR_HISTORY_SUMMARY " + json.dumps({
    "revisions": len(revisions), "blobs": len(blobs), "unique_methods": len(unique_methods),
    "method_variants": unique_methods
}), flush=True)
