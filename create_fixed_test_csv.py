from data import create_fixed_test_csv
from pathlib import Path

if __name__ == "__main__":
    create_fixed_test_csv(
        Path("student_start_pack/test_5000_samples.csv"),
        num_samples=5000,
        seed=42
    )