import os
from pathlib import Path


def generate_50_mock():
    tmp_dir = Path("/home/kymy/Documents/python/protein_predict/tmp_data")
    tmp_dir.mkdir(exist_ok=True)

    # 1. FASTA files (train and testsuperset)
    train_fasta = []
    test_fasta = []
    for i in range(1, 51):
        pid = f"P{i:05d}"
        seq = "ACDEFGHIKLMNPQRSTVWY" * ((i % 5) + 1)
        train_fasta.append(f">sp|{pid}|ID{i}\n{seq}\n")

        tid = f"T{i:05d}"
        test_fasta.append(f">sp|{tid}|ID{i}\n{seq}\n")

    with open(tmp_dir / "mock_train_sequences.fasta", "w") as f:
        f.writelines(train_fasta)
    with open(tmp_dir / "mock_testsuperset.fasta", "w") as f:
        f.writelines(test_fasta)

    # 2. Taxonomy files
    train_tax = []
    test_tax = []
    for i in range(1, 51):
        pid = f"P{i:05d}"
        taxon = 1000 + (i % 5)
        train_tax.append(f"{pid}\t{taxon}\n")

        tid = f"T{i:05d}"
        test_tax.append(f"{tid}\t{taxon}\n")

    with open(tmp_dir / "mock_train_taxonomy.tsv", "w") as f:
        f.writelines(train_tax)
    with open(tmp_dir / "mock_test_taxonomy.tsv", "w") as f:
        f.writelines(test_tax)

    # 3. Terms file (train_terms)
    terms = ["GO:0005575", "GO:0005576", "GO:0005577"]
    train_terms = ["EntryID\tterm\n"]
    for i in range(1, 51):
        pid = f"P{i:05d}"
        # Give each protein 1-2 terms
        train_terms.append(f"{pid}\t{terms[0]}\n")
        if i % 2 == 0:
            train_terms.append(f"{pid}\t{terms[i % 2 + 1]}\n")

    with open(tmp_dir / "mock_train_terms.tsv", "w") as f:
        f.writelines(train_terms)

    print("Generated 50 mock entries for all files in tmp_data.")


if __name__ == "__main__":
    generate_50_mock()
