#!/usr/bin/env python3
"""
GATv2 Knowledge Graph Visualizer
=================================
Generates interactive 2D and 3D visualizations of the waste-classification
knowledge graph, including GNN-learned node embeddings projected via t-SNE.

Outputs:
  1. gnn_knowledge_graph_2d.html  — interactive 2D graph (Plotly)
  2. gnn_knowledge_graph_3d.html  — interactive 3D graph (Plotly)
  3. gnn_embedding_space.html     — 3D t-SNE of GATv2 embeddings (Plotly)

Usage:
    python scripts/visualize_gnn.py
"""

import sys, os, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import networkx as nx
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# ── Knowledge graph definition (mirrors run_gnn_training.py exactly) ──
TARGET_CLASSES = [
    'aerosol_cans','aluminum_food_cans','aluminum_soda_cans','cardboard_boxes',
    'cardboard_packaging','clothing','coffee_grounds','disposable_plastic_cutlery',
    'eggshells','food_waste','glass_beverage_bottles','glass_cosmetic_containers',
    'glass_food_jars','magazines','newspaper','office_paper','paper_cups',
    'plastic_cup_lids','plastic_detergent_bottles','plastic_food_containers',
    'plastic_shopping_bags','plastic_soda_bottles','plastic_straws',
    'plastic_trash_bags','plastic_water_bottles','shoes','steel_food_cans',
    'styrofoam_cups','styrofoam_food_containers','tea_bags',
]
MATERIALS = ['plastic','paper','glass','metal','organic','textile','styrofoam','mixed']
BINS = ['recycle','compost','landfill','special','donate']

ITEM_MATERIAL = {
    'aerosol_cans':'metal','aluminum_food_cans':'metal','aluminum_soda_cans':'metal',
    'steel_food_cans':'metal','cardboard_boxes':'paper','cardboard_packaging':'paper',
    'magazines':'paper','newspaper':'paper','office_paper':'paper','paper_cups':'paper',
    'glass_beverage_bottles':'glass','glass_cosmetic_containers':'glass',
    'glass_food_jars':'glass','disposable_plastic_cutlery':'plastic',
    'plastic_cup_lids':'plastic','plastic_detergent_bottles':'plastic',
    'plastic_food_containers':'plastic','plastic_shopping_bags':'plastic',
    'plastic_soda_bottles':'plastic','plastic_straws':'plastic',
    'plastic_trash_bags':'plastic','plastic_water_bottles':'plastic',
    'coffee_grounds':'organic','eggshells':'organic','food_waste':'organic',
    'tea_bags':'organic','clothing':'textile','shoes':'mixed',
    'styrofoam_cups':'styrofoam','styrofoam_food_containers':'styrofoam',
}
MATERIAL_BIN = {
    'plastic':'recycle','paper':'recycle','glass':'recycle','metal':'recycle',
    'organic':'compost','textile':'donate','styrofoam':'landfill','mixed':'special',
}
ITEM_BIN_OVERRIDE = {
    'disposable_plastic_cutlery':'landfill','plastic_straws':'landfill',
    'plastic_trash_bags':'landfill','plastic_shopping_bags':'special',
    'paper_cups':'landfill','shoes':'donate',
}
CONFUSION_PAIRS = [
    ('glass_beverage_bottles','glass_food_jars'),('cardboard_boxes','cardboard_packaging'),
    ('aluminum_food_cans','steel_food_cans'),('aluminum_soda_cans','steel_food_cans'),
    ('plastic_soda_bottles','plastic_water_bottles'),
    ('plastic_soda_bottles','plastic_detergent_bottles'),
    ('newspaper','office_paper'),('newspaper','magazines'),('office_paper','magazines'),
    ('styrofoam_cups','styrofoam_food_containers'),
    ('plastic_food_containers','plastic_cup_lids'),
    ('plastic_food_containers','disposable_plastic_cutlery'),
    ('coffee_grounds','tea_bags'),('food_waste','coffee_grounds'),('clothing','shoes'),
]

# ── colour palette ──
MAT_COLOURS = {
    'plastic':'#e74c3c','paper':'#f39c12','glass':'#2ecc71','metal':'#3498db',
    'organic':'#27ae60','textile':'#9b59b6','styrofoam':'#e67e22','mixed':'#95a5a6',
}
BIN_COLOURS = {'recycle':'#1abc9c','compost':'#2ecc71','landfill':'#e74c3c','special':'#f39c12','donate':'#9b59b6'}
NODE_TYPE_SHAPE = {'Item':'circle','Material':'diamond','Bin':'square'}

def build_graph():
    """Build NetworkX graph with full metadata."""
    G = nx.DiGraph()
    num_cls = len(TARGET_CLASSES)
    mat_base = num_cls
    bin_base = mat_base + len(MATERIALS)

    # Add item nodes
    for i, cls in enumerate(TARGET_CLASSES):
        mat = ITEM_MATERIAL.get(cls, 'mixed')
        G.add_node(i, name=cls.replace('_',' ').title(), node_type='Item',
                   material=mat, color=MAT_COLOURS[mat], size=12)
    # Material nodes
    for i, m in enumerate(MATERIALS):
        G.add_node(mat_base+i, name=f"⚗️ {m.upper()}", node_type='Material',
                   material=m, color=MAT_COLOURS[m], size=25)
    # Bin nodes
    for i, b in enumerate(BINS):
        G.add_node(bin_base+i, name=f"🗑️ {b.upper()}", node_type='Bin',
                   material='bin', color=BIN_COLOURS[b], size=30)

    mat_idx = {m: mat_base+i for i,m in enumerate(MATERIALS)}
    bin_idx = {b: bin_base+i for i,b in enumerate(BINS)}

    for i, cls in enumerate(TARGET_CLASSES):
        mat = ITEM_MATERIAL.get(cls, 'mixed')
        G.add_edge(i, mat_idx[mat], edge_type='MADE_OF', color='#aaa')
    for mat, b in MATERIAL_BIN.items():
        G.add_edge(mat_idx[mat], bin_idx[b], edge_type='GOES_TO', color='#888')
    for cls, b in ITEM_BIN_OVERRIDE.items():
        G.add_edge(TARGET_CLASSES.index(cls), bin_idx[b], edge_type='OVERRIDE', color='#e74c3c')
    for a, b in CONFUSION_PAIRS:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        G.add_edge(ia, ib, edge_type='SIMILAR_TO', color='#f1c40f')
        G.add_edge(ib, ia, edge_type='SIMILAR_TO', color='#f1c40f')
    return G

def compute_gnn_embeddings():
    """Run GATv2 forward pass and return 3D t-SNE coords + raw embeddings."""
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data as PyGData
    import torch.nn as nn

    print("  Building knowledge graph tensor data...")
    torch.manual_seed(42)
    feat_dim = 128
    num_cls = len(TARGET_CLASSES)
    mat_base, bin_base = num_cls, num_cls + len(MATERIALS)
    total = num_cls + len(MATERIALS) + len(BINS)  # 43

    # Build feature matrix
    x = torch.randn(total, feat_dim)
    for i, mat in enumerate(MATERIALS):
        x[mat_base+i] = torch.randn(feat_dim) * 0.5
        s = i * (feat_dim // len(MATERIALS))
        e = (i+1) * (feat_dim // len(MATERIALS))
        x[mat_base+i, s:e] += 2.0
    for i, b in enumerate(BINS):
        x[bin_base+i] = torch.randn(feat_dim) * 0.3
        s = i * (feat_dim // len(BINS))
        e = (i+1) * (feat_dim // len(BINS))
        x[bin_base+i, s:e] += 3.0

    # Build edge index
    mat_idx = {m: mat_base+i for i,m in enumerate(MATERIALS)}
    bin_idx = {b: bin_base+i for i,b in enumerate(BINS)}
    src, tgt = [], []
    for i, cls in enumerate(TARGET_CLASSES):
        mat = ITEM_MATERIAL.get(cls, 'mixed')
        m = mat_idx[mat]; src += [i, m]; tgt += [m, i]
    for mat, b in MATERIAL_BIN.items():
        m = mat_idx[mat]; bi = bin_idx[b]; src += [m, bi]; tgt += [bi, m]
    for cls, b in ITEM_BIN_OVERRIDE.items():
        bi = bin_idx[b]; ii = TARGET_CLASSES.index(cls); src += [ii, bi]; tgt += [bi, ii]
    for a, b in CONFUSION_PAIRS:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        src += [ia, ib]; tgt += [ib, ia]
    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    # Build GATv2 model (matches run_gnn_training.py)
    class GATv2Vis(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATv2Conv(feat_dim, 64, heads=4, concat=True, dropout=0.2)
            self.conv2 = GATv2Conv(64*4, 64, heads=4, concat=True, dropout=0.2)
            self.conv3 = GATv2Conv(64*4, 64, heads=1, concat=False, dropout=0.2)
        def forward(self, x, ei):
            x = F.elu(self.conv1(x, ei))
            x = F.elu(self.conv2(x, ei))
            x = self.conv3(x, ei)
            return F.normalize(x, p=2, dim=-1)

    model = GATv2Vis()
    model.eval()
    print(f"  GATv2 model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Try loading trained checkpoint
    ckpt_paths = list(Path("models/gnn").rglob("*.pth"))
    loaded = False
    for cp in ckpt_paths:
        try:
            ckpt = torch.load(cp, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(sd, strict=False)
            print(f"  ✅ Loaded trained weights from {cp}")
            loaded = True
            break
        except Exception:
            continue
    if not loaded:
        print("  ⚠️  No trained checkpoint found — using random init (structure is correct)")

    with torch.no_grad():
        embeddings = model(x, edge_index)  # (43, 64)

    # t-SNE → 3D
    print("  Running t-SNE (3D projection of 64-d embeddings)...")
    tsne = TSNE(n_components=3, perplexity=min(15, total-1), random_state=42, n_iter=2000)
    coords_3d = tsne.fit_transform(embeddings.numpy())

    return coords_3d, embeddings.numpy()


# ═══════════════════════════════════════════════════════════════════
#  PLOTLY RENDERERS
# ═══════════════════════════════════════════════════════════════════

def render_2d_graph(G, output):
    """Interactive 2D force-directed graph."""
    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42)

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=1, color=d.get('color','#aaa')),
            hoverinfo='text', text=d.get('edge_type',''),
            showlegend=False,
        ))

    # Separate traces per node type for legend
    node_traces = {}
    for nid, d in G.nodes(data=True):
        nt = d['node_type']
        if nt not in node_traces:
            node_traces[nt] = dict(x=[], y=[], text=[], color=[], size=[])
        x, y = pos[nid]
        node_traces[nt]['x'].append(x)
        node_traces[nt]['y'].append(y)
        node_traces[nt]['text'].append(f"{d['name']}<br>Type: {nt}<br>Material: {d.get('material','')}")
        node_traces[nt]['color'].append(d['color'])
        node_traces[nt]['size'].append(d['size'])

    scatter_traces = []
    symbols = {'Item':'circle','Material':'diamond','Bin':'square'}
    for nt, vals in node_traces.items():
        scatter_traces.append(go.Scatter(
            x=vals['x'], y=vals['y'], mode='markers+text',
            marker=dict(size=vals['size'], color=vals['color'],
                        symbol=symbols.get(nt,'circle'),
                        line=dict(width=1, color='white')),
            text=[t.split('<br>')[0] for t in vals['text']],
            textposition='top center', textfont=dict(size=7),
            hovertext=vals['text'], hoverinfo='text', name=nt,
        ))

    fig = go.Figure(data=edge_traces + scatter_traces)
    fig.update_layout(
        title='ReLEAF AI — GATv2 Knowledge Graph (2D)',
        template='plotly_dark', width=1400, height=900,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(font=dict(size=14)),
    )
    fig.write_html(output, auto_open=False)
    print(f"  ✅ Saved → {output}")


def render_3d_graph(G, output):
    """Interactive 3D force-directed graph."""
    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42, dim=3)

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
        edge_traces.append(go.Scatter3d(
            x=[x0,x1,None], y=[y0,y1,None], z=[z0,z1,None],
            mode='lines', line=dict(width=1.5, color=d.get('color','#aaa')),
            hoverinfo='text', text=d.get('edge_type',''), showlegend=False,
        ))

    scatter_traces = []
    symbols3d = {'Item':'circle','Material':'diamond','Bin':'square'}
    node_groups = {}
    for nid, d in G.nodes(data=True):
        nt = d['node_type']
        if nt not in node_groups:
            node_groups[nt] = dict(x=[],y=[],z=[],text=[],color=[],size=[])
        x,y,z = pos[nid]
        node_groups[nt]['x'].append(x)
        node_groups[nt]['y'].append(y)
        node_groups[nt]['z'].append(z)
        node_groups[nt]['text'].append(f"{d['name']}<br>Type: {nt}<br>Material: {d.get('material','')}")
        node_groups[nt]['color'].append(d['color'])
        node_groups[nt]['size'].append(d['size'])

    for nt, v in node_groups.items():
        scatter_traces.append(go.Scatter3d(
            x=v['x'], y=v['y'], z=v['z'], mode='markers+text',
            marker=dict(size=[s*0.5 for s in v['size']], color=v['color'],
                        symbol=symbols3d.get(nt,'circle'),
                        line=dict(width=0.5, color='white')),
            text=[t.split('<br>')[0] for t in v['text']],
            textfont=dict(size=7),
            hovertext=v['text'], hoverinfo='text', name=nt,
        ))

    fig = go.Figure(data=edge_traces + scatter_traces)
    fig.update_layout(
        title='ReLEAF AI — GATv2 Knowledge Graph (3D)',
        template='plotly_dark', width=1400, height=900,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='#111111',
        ),
        legend=dict(font=dict(size=14)),
    )
    fig.write_html(output, auto_open=False)
    print(f"  ✅ Saved → {output}")


def render_embedding_space(G, coords_3d, embeddings, output):
    """3D t-SNE of GATv2 node embeddings — shows learned vector space."""
    nodes_list = sorted(G.nodes(data=True))

    # One trace per node type
    traces = []
    for nt in ['Item', 'Material', 'Bin']:
        idxs = [i for i,(nid,d) in enumerate(nodes_list) if d['node_type']==nt]
        if not idxs:
            continue
        traces.append(go.Scatter3d(
            x=[coords_3d[i,0] for i in idxs],
            y=[coords_3d[i,1] for i in idxs],
            z=[coords_3d[i,2] for i in idxs],
            mode='markers+text',
            marker=dict(
                size=[nodes_list[i][1]['size']*0.4 for i in idxs],
                color=[nodes_list[i][1]['color'] for i in idxs],
                symbol={'Item':'circle','Material':'diamond','Bin':'square'}[nt],
                line=dict(width=0.5, color='white'),
                opacity=0.9,
            ),
            text=[nodes_list[i][1]['name'] for i in idxs],
            textfont=dict(size=7, color='white'),
            hovertext=[
                f"{nodes_list[i][1]['name']}<br>"
                f"Type: {nt}<br>"
                f"Embedding norm: {np.linalg.norm(embeddings[i]):.3f}<br>"
                f"t-SNE: ({coords_3d[i,0]:.1f}, {coords_3d[i,1]:.1f}, {coords_3d[i,2]:.1f})"
                for i in idxs
            ],
            hoverinfo='text', name=nt,
        ))

    # Draw lines between nodes connected by edges
    edge_trace = []
    for u, v, d in G.edges(data=True):
        x0,y0,z0 = coords_3d[u]
        x1,y1,z1 = coords_3d[v]
        edge_trace.append(go.Scatter3d(
            x=[x0,x1,None], y=[y0,y1,None], z=[z0,z1,None],
            mode='lines', line=dict(width=1, color='rgba(150,150,150,0.2)'),
            showlegend=False, hoverinfo='none',
        ))

    fig = go.Figure(data=edge_trace + traces)
    fig.update_layout(
        title='ReLEAF AI — GATv2 Embedding Space (t-SNE 3D projection of 64-d vectors)',
        template='plotly_dark', width=1400, height=900,
        scene=dict(
            xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', zaxis_title='t-SNE 3',
            bgcolor='#111111',
        ),
        legend=dict(font=dict(size=14)),
    )
    fig.write_html(output, auto_open=False)
    print(f"  ✅ Saved → {output}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = PROJECT_ROOT / 'outputs' / 'gnn_viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ReLEAF AI — GATv2 Knowledge Graph Visualizer              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n1️⃣  Building knowledge graph...")
    G = build_graph()
    print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n2️⃣  Computing GATv2 embeddings...")
    coords_3d, embeddings = compute_gnn_embeddings()

    print("\n3️⃣  Rendering 2D graph...")
    render_2d_graph(G, str(out_dir / 'gnn_knowledge_graph_2d.html'))

    print("\n4️⃣  Rendering 3D graph...")
    render_3d_graph(G, str(out_dir / 'gnn_knowledge_graph_3d.html'))

    print("\n5️⃣  Rendering embedding space (t-SNE 3D)...")
    render_embedding_space(G, coords_3d, embeddings, str(out_dir / 'gnn_embedding_space.html'))

    print(f"\n✅ All visualizations saved to {out_dir}/")
    print("   Open any .html file in your browser to interact (pan, zoom, hover).")
    print(f"\n   open {out_dir / 'gnn_knowledge_graph_3d.html'}")
    print(f"   open {out_dir / 'gnn_embedding_space.html'}")
