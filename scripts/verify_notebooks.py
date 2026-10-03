"""Audit and verify generated Jupyter notebooks."""

import ast
import json
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"

for nb_name in ["FedTROS_Colab_Runner.ipynb", "FedTROS_Colab_Drive_Runner.ipynb"]:
    path = DEPLOY_DIR / nb_name
    assert path.is_file(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\n{'='*75}")
    print(f"VERIFYING: {nb_name} (Total cells: {len(nb['cells'])})")
    print(f"{'='*75}")

    code_cells = 0
    md_cells = 0
    errors = 0

    for idx, cell in enumerate(nb["cells"]):
        ctype = cell["cell_type"]
        src = "".join(cell.get("source", []))
        first_line = src.strip().split("\n")[0][:75] if src.strip() else "<EMPTY>"

        if ctype == "code":
            code_cells += 1
            # Filter out bash magic lines (!, %)
            clean_lines = []
            for line in src.split("\n"):
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_src = "\n".join(clean_lines)

            try:
                ast.parse(clean_src)
                status = "PASS"
            except SyntaxError as e:
                status = f"FAIL: {e.msg} (line {e.lineno})"
                errors += 1

            # Clean leading hashes and dividers
            title = "<NO TITLE>"
            for l in src.split("\n"):
                l_str = l.strip()
                if l_str and not l_str.startswith("# ==") and not l_str.startswith("=="):
                    title = l_str
                    break
            print(f"  [{idx:02d}] CODE ({status:<5}): {title[:75]}")
        else:
            md_cells += 1
            print(f"  [{idx:02d}] MD         : {first_line[:75]}")

    print(f"Summary: Code Cells={code_cells}, Markdown Cells={md_cells}, Errors={errors}")
    assert errors == 0, f"Errors found in {nb_name}!"
    print(f"[OK] {nb_name} is 100% syntactically valid and structurally complete.")
