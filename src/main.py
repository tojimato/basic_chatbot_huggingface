"""Basit çalıştırılabilir giriş noktası."""

import os
from pathlib import Path
from dotenv import load_dotenv


def main():
    """Run a simple demo action and load local env."""
    # Load .env.local from repository root
    env_path = Path(__file__).resolve().parent.parent / ".env.local"
    load_dotenv(dotenv_path=env_path)

    sample = os.getenv("SAMPLE_VAR", "<not set>")

    print("Merhaba! Proje çalışıyor.")
    print(f"SAMPLE_VAR={sample}")


if __name__ == "__main__":
    main()
