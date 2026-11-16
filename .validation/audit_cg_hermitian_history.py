import ast
import json
import subprocess

def git(*args):
    return subprocess.check_output(["git", *args], text=True)

subprocess.run(["git", "fetch", "--no-tags", "https://github.com/Afloat16/pypose.git",
                "+refs/heads/*:refs/remotes/history-fork/*"], check=True)
subprocess.run(["git", "fetch", "--no-tags", "https://github.com/pypose/pypose.git",
                "+refs/heads/*:refs/remotes/history-upstream/*",
                "+refs/pull/*/head:refs/remotes/history-pr/*"], check=True)
paths = {"pypose/optim/solver.py"}
renames = []
for line in git("log", "--all", "--format=", "--name-status", "--find-renames").splitlines():
    fields = line.split("\t")
    if len(fields) == 3 and fields[0].startswith("R"):
        renames.append(fields[1:])
changed = True
while changed:
    changed = False
    for left, right in renames:
        if left in paths or right in paths:
            for name in (left, right):
                if name not in paths:
                    paths.add(name)
                    changed = True
objects = {}
for line in git("rev-list", "--objects", "--all", "--", *sorted(paths)).splitlines():
    fields = line.split(" ", 1)
    if len(fields) == 2 and fields[1] in paths:
        objects.setdefault(fields[0], []).append(fields[1])
blobs = []
for sha, names in objects.items():
    if git("cat-file", "-t", sha).strip() != "blob":
        continue
    source = git("cat-file", "blob", sha)
    try:
        module = ast.parse(source)
    except SyntaxError:
        blobs.append({"sha": sha, "paths": names, "parse_failed": True, "source": source})
        continue
    cg = [ast.get_source_segment(source, node) for node in module.body if isinstance(node, ast.ClassDef) and node.name == "CG"]
    if cg:
        blobs.append({"sha": sha, "paths": names, "class_source": cg[0]})
refs = git("for-each-ref", "--format=%(refname)").splitlines()
report = {
    "base": "298e9f0862cb228b0e8851b2616e682ccdca6a32",
    "refs": refs, "ref_count": len(refs),
    "fork_branch_count": sum(ref.startswith("refs/remotes/history-fork/") for ref in refs),
    "upstream_pr_head_count": sum(ref.startswith("refs/remotes/history-pr/") for ref in refs),
    "reachable_commit_count": int(git("rev-list", "--count", "--all")),
    "paths": sorted(paths), "rename_edges": [r for r in renames if r[0] in paths or r[1] in paths],
    "source_blob_count": len(objects), "cg_blob_count": len(blobs), "blobs": blobs,
}
print("PYPOSE_CG_HISTORY_BEGIN")
print(json.dumps(report))
print("PYPOSE_CG_HISTORY_END")
