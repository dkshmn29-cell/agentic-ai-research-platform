from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
)


def main() -> None:
    downloaded_directory = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIRECTORY,
    )
    print(f"Downloaded {MODEL_ID} to {downloaded_directory}")
    print("Copy the all-MiniLM-L6-v2 directory to the offline project models folder.")


if __name__ == "__main__":
    main()