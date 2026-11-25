"""Publish the runtime files to a free Gradio ZeroGPU Space."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware
from huggingface_hub.utils import get_stored_tokens

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="zaabinh/aspectai")
    parser.add_argument("--token-name", help="Name of an existing saved Hugging Face login")
    args = parser.parse_args()
    token = get_stored_tokens()[args.token_name] if args.token_name else None
    api = HfApi(token=token)
    print(api.create_repo(
        args.repo, repo_type="space", space_sdk="gradio",
        space_hardware=SpaceHardware.ZERO_A10G, private=False, exist_ok=True,
    ), flush=True)
    runtime = api.get_space_runtime(args.repo)
    if (runtime.requested_hardware or runtime.hardware) != "zero-a10g":
        raise RuntimeError("Existing Space is not configured for free ZeroGPU; upload stopped.")

    files = {Path(p) for p in [
        "README.md", "preprocessing.py", "manual_ovr_logistic_regression.py", "tfidf_helper.py",
        "demo/space.py", "demo/space_ui.py", "demo/space.css", "demo/inference.py", "demo/requirements.txt",
        "outputs/final_comparison/final_model_comparison.csv",
        "outputs/EDA/aspect_frequency.png",
    ]}
    for folder in ["models/mpnet", "models/TF-IDF", "docs"]:
        files.update(p.relative_to(ROOT) for p in (ROOT / folder).rglob("*") if p.is_file())
    for relative in files:
        if relative.suffix in {".safetensors", ".joblib"}:
            with (ROOT / relative).open("rb") as f:
                if f.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise RuntimeError(f"Run git lfs pull before uploading {relative}")
    operations = [
        CommitOperationAdd(path_in_repo=p.as_posix(), path_or_fileobj=ROOT / p)
        for p in sorted(files)
    ]
    operations.append(CommitOperationAdd(
        path_in_repo="requirements.txt", path_or_fileobj=ROOT / "deploy/zerogpu/requirements.txt",
    ))
    info = api.space_info(args.repo, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    changed = []
    for operation in operations:
        previous = remote.get(operation.path_in_repo)
        path = Path(operation.path_or_fileobj)
        if previous is not None:
            if previous.lfs:
                with path.open("rb") as f:
                    identical = hashlib.file_digest(f, "sha256").hexdigest() == previous.lfs.sha256
            else:
                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                identical = blob == previous.blob_id
            if identical:
                continue
        changed.append(operation)
    if not changed:
        print("Space is already up to date.", flush=True)
        return
    print(f"Uploading {len(changed)} changed files.", flush=True)
    result = api.create_commit(
        repo_id=args.repo, repo_type="space", operations=changed, parent_commit=info.sha,
        commit_message="Update AspectAI model selection and Gradio interface",
    )
    print(result.commit_url, flush=True)


if __name__ == "__main__":
    main()
