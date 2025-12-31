import json

with open("Pretraitement.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

mappings = {}
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output['output_type'] == 'stream':
                text = "".join(output['text'])
                if "Légende pour" in text:
                    lines = text.split('\n')
                    current_col = None
                    for line in lines:
                        if "Légende pour" in line:
                            current_col = line.split("Légende pour")[1].strip().replace(":", "")
                            mappings[current_col] = {}
                        elif "→" in line and current_col:
                            parts = line.split("→")
                            code = int(parts[0].strip())
                            label = parts[1].strip()
                            mappings[current_col][code] = label

print(json.dumps(mappings))
