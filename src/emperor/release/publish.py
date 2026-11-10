"""Upload weights directories to the Hugging Face Hub, only when explicitly confirmed.

The token is never an argument: `huggingface_hub` reads it from the `HF_TOKEN` environment
variable or the login cache (`hf auth login`). Only model files are uploaded.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
ALLOW = ["*.safetensors", "*.gguf", "*.json", "*.md", "*.txt", "*.jinja", "*.model"]


class PublishRefused(RuntimeError):
    """Publishing was refused (not confirmed, or something is not ready)."""


@dataclass
class Upload:
    repo_id: str
    folder: Path


def check_ready(uploads: list[Upload]) -> None:
    if not uploads:
        raise PublishRefused("nothing to publish: pass --repo-id and/or --gguf-repo-id")
    for upload in uploads:
        if not REPO_ID.match(upload.repo_id):
            raise PublishRefused(f"{upload.repo_id!r} is not an owner/name repo id")
        if not upload.folder.is_dir():
            raise PublishRefused(f"{upload.folder} does not exist")
        if not (upload.folder / "README.md").is_file():
            raise PublishRefused(f"{upload.folder} has no README.md; run the card step first")


def publish(
    uploads: list[Upload],
    *,
    confirm: bool,
    private: bool = True,
    api_factory: Callable[[], object] | None = None,
) -> list[str]:
    """Create each repo (private unless `private` is False) and upload its folder."""
    check_ready(uploads)
    if not confirm:
        raise PublishRefused(
            "refusing to publish without --confirm; would upload "
            + ", ".join(f"{u.folder} -> {u.repo_id}" for u in uploads)
        )
    if api_factory is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise PublishRefused(
                "huggingface_hub is not installed; use `uv run --group publish release publish`"
            ) from error
        api_factory = HfApi
    api = api_factory()
    urls = []
    for upload in uploads:
        api.create_repo(upload.repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=upload.repo_id,
            repo_type="model",
            folder_path=str(upload.folder),
            allow_patterns=ALLOW,
            commit_message="Upload model files",
        )
        urls.append(f"https://huggingface.co/{upload.repo_id}")
    return urls
