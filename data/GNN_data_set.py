
# =========================================
# MatPES → Graph Dataset Builder (FIXED)
# Handles Decimal errors + scalable to 300k
# =========================================

import os
import ijson
import numpy as np
import torch
from tqdm import tqdm
from decimal import Decimal
from pymatgen.core import Structure
from torch_geometric.data import Data


# =========================================
# 1. FIX: Convert Decimal → float safely
# =========================================
def clean_decimals(obj):

    if isinstance(obj, dict):
        return {k: clean_decimals(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [clean_decimals(x) for x in obj]

    elif isinstance(obj, Decimal):
        return float(obj)

    else:
        return obj


# =========================================
# 2. Graph edge builder (cutoff graph)
# =========================================
def build_edges(pos, cutoff=5.0):

    row, col = [], []
    edge_attr = []

    N = len(pos)

    for i in range(N):
        for j in range(i + 1, N):

            dist = np.linalg.norm(pos[i] - pos[j])

            if dist < cutoff:

                vec = pos[j] - pos[i]

                row += [i, j]
                col += [j, i]

                edge_attr += [
                    [dist, vec[0], vec[1], vec[2]],
                    [dist, -vec[0], -vec[1], -vec[2]]
                ]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


# =========================================
# 3. Structure → Graph
# =========================================
def structure_to_graph(structure, label, cutoff=5.0):

    Z = []
    pos = []

    for site in structure:
        Z.append(site.specie.Z)
        pos.append(site.coords)

    x = torch.tensor(Z, dtype=torch.float).view(-1, 1)
    pos = np.array(pos)

    edge_index, edge_attr = build_edges(pos, cutoff)

    return Data(
        x=x,
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long)
    )


# =========================================
# 4. Output folder
# =========================================
output_dir = "graph_dataset"
os.makedirs(output_dir, exist_ok=True)


# =========================================
# 5. MAIN PIPELINE (STREAMING + SAFE)
# =========================================
def build_graph_dataset(json_file, max_entries=None, cutoff=5.0):

    print("\n🚀 Starting graph dataset construction...\n")

    with open(json_file, "rb") as f:

        items = ijson.items(f, "item")

        for i, row in enumerate(tqdm(items)):

            if max_entries and i >= max_entries:
                break

            try:
                # -----------------------------
                # FIX: Decimal cleaning step
                # -----------------------------
                row = clean_decimals(row)

                structure_dict = row["structure"]

                structure = Structure.from_dict(structure_dict)

                # Label (metal / insulator)
                bandgap = row.get("bandgap", 0)
                label = 1 if bandgap < 0.1 else 0

                graph = structure_to_graph(structure, label, cutoff)

                torch.save(
                    graph,
                    os.path.join(output_dir, f"data_{i}.pt")
                )

            except Exception as e:
                print(f"❌ Skip {i}: {e}")

            if i % 1000 == 0:
                print(f"✔ Processed {i} structures")


    print("\n=================================")
    print("✅ GRAPH DATASET CREATED")
    print("📁 Folder:", output_dir)
    print("=================================\n")


# =========================================
# 6. RUN SCRIPT
# =========================================
if __name__ == "__main__":

    json_file = "MatPES-R2SCAN-2025.2.json"  # CHANGE PATH IF NEEDED

    build_graph_dataset(
        json_file=json_file,
        max_entries=None,   # FULL 300k dataset
        cutoff=5.0
    )
