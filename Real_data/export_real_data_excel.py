from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from Real_data.real_data_export import RealDataExporter


def main():
    input_path = BASE_DIR / "Real_data.xlsx"
    output_path = BASE_DIR / "real_data_smiles_target.xlsx"

    exporter = RealDataExporter(input_path)
    exporter.export_excel(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
