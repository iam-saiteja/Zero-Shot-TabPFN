import json
import shutil
from pathlib import Path

def main():
    # Base paths
    base_dir = Path(__file__).resolve().parent.parent / "tabzilla" / "TabZilla"
    datasets_dir = base_dir / "datasets"
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found at {datasets_dir}")
        return

    # Create category directories
    categories = {
        "small": datasets_dir / "small",    # < 10k rows
        "medium": datasets_dir / "medium",  # 10k - 100k rows
        "large": datasets_dir / "large"     # > 100k rows
    }
    
    for path in categories.values():
        path.mkdir(exist_ok=True)

    # Scan and organize
    moved_count = 0
    for d_path in datasets_dir.iterdir():
        # Skip if it's not a directory or if it's one of our category folders
        if not d_path.is_dir() or d_path.name in categories.keys():
            continue
            
        metadata_file = d_path / "metadata.json"
        if not metadata_file.exists():
            continue
            
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        num_instances = metadata.get("num_instances", 0)
        
        # Determine category
        if num_instances < 10000:
            target_cat = "small"
        elif num_instances <= 100000:
            target_cat = "medium"
        else:
            target_cat = "large"
            
        dest_path = categories[target_cat] / d_path.name
        
        print(f"Moving {d_path.name} ({num_instances} rows) -> {target_cat}/")
        shutil.move(str(d_path), str(dest_path))
        moved_count += 1
        
    print(f"\nOrganization complete! Moved {moved_count} datasets into TabArena size splits.")

if __name__ == "__main__":
    main()
