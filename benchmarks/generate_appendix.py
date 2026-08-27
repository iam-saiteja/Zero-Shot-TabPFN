import json

with open('benchmarks/tfm_leaderboard.json') as f:
    data = json.load(f)

ds_set = set()
for row in data:
    ds = row['dataset']
    # e.g., 'small/openml__page-blocks__30' -> 'page-blocks (30)'
    parts = ds.split('/')[-1].split('__')
    if len(parts) >= 3:
        clean = f"{parts[1].replace('_', r'\_')} (ID: {parts[2]})"
    else:
        clean = ds.split('/')[-1].replace('_', r'\_')
    ds_set.add(clean)

datasets = sorted(list(ds_set))

with open('benchmarks/appendix.tex', 'w') as out:
    out.write("\\section{Appendix A: Evaluated Datasets}\n\n")
    out.write("The following 186 datasets from the TabZilla benchmark suite were evaluated in this study. Datasets are listed alphabetically.\n\n")
    out.write("\\begin{itemize}\n")
    out.write("  \\setlength\\itemsep{0em}\n")
    for ds in datasets:
        out.write(f"  \\item {ds}\n")
    out.write("\\end{itemize}\n")
    
print("Appendix written to benchmarks/appendix.tex")
