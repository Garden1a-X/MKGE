# MKGE

Official code for **Grounding User Preferences in Shared Multimodal Concepts for Recommendation**, accepted to **WSDM 2027 (Main Conference)**.

Ao Xu, Jinhui Huang, Wenqiang Tang, Tiehua Mei, Hongru Hou, Hengrui Chen, Jiaqing Liang, and Deqing Yang.

Paper DOI: [10.1145/3862322.3863599](https://doi.org/10.1145/3862322.3863599). The DOI will become active after publication.

Contact: [Ao Xu](mailto:axu24@m.fudan.edu.cn).

This release carries forward the implementation, configurations, and prompt templates from the review repository. See [data/README.md](data/README.md) to download the released interactions and constructed knowledge graphs before running the training commands below.

## Repository Structure

```
├── src/
│   ├── data/
│   │   ├── graph_builder.py    # Heterogeneous graph construction (User-Entity-Item)
│   │   └── dataset.py          # Data splitting and DataLoader creation
│   ├── model/
│   │   ├── ours.py             # Main model: KnowledgeEnhancedRecModel
│   │   ├── encoders.py         # CF encoder (LightGCN) and KG encoder (HeteroGAT)
│   │   └── losses.py           # InfoNCE, BPR, contrastive, alignment, mask losses
│   └── utils/
│       ├── config.py           # Configuration loading
│       └── metrics.py          # NDCG, Recall, Precision, Hit evaluation
├── scripts/
│   └── train_model.py          # Training and evaluation script
├── configs/
│   ├── ml1m.yaml               # ML-1M configuration
│   ├── beauty.yaml             # Amazon Beauty configuration
│   └── videogames.yaml         # Amazon Video Games configuration
└── data/
    ├── ml-1m/                  # ML-1M dataset (5-core filtered)
    │   ├── ml-1m.inter         # User-item interactions
    │   ├── ml-1m.item.kg       # Item visual knowledge graph
    │   └── ml-1m.user.kg       # User interest knowledge graph
    ├── amazon-beauty/          # Amazon Beauty dataset
    │   ├── amazon-beauty.inter
    │   ├── amazon-beauty.item.kg
    │   └── amazon-beauty.user.kg
    └── amazon-videogames/      # Amazon Video Games dataset (5-core filtered)
        ├── amazon-videogames.inter
        ├── amazon-videogames.item.kg
        └── amazon-videogames.user.kg
```

## Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4

## Training

```bash
# ML-1M
python scripts/train_model.py --config configs/ml1m.yaml

# Amazon Beauty
python scripts/train_model.py --config configs/beauty.yaml

# Amazon Video Games
python scripts/train_model.py --config configs/videogames.yaml
```

## Evaluation

Evaluation is performed automatically during training using the **uni100** protocol (1 positive + 99 random negatives per user). Metrics reported: NDCG@{5,10,20} and Recall@{5,10,20}.

## Data Format

All data files use RecBole's tab-separated format:

**Interaction file** (`.inter`):
```
user_id:token    item_id:token    rating:float    timestamp:float
```

**Knowledge graph file** (`.item.kg` / `.user.kg`):
```
head_id:token    relation_id:token    tail_id:token
```

- **Item KG**: `head_id` = item ID, `tail_id` = visual entity (e.g., `dramatic_lighting`, `action_scenes`)
- **User KG**: `head_id` = user ID, `relation_id` ∈ {`long_term_interest`, `short_term_interest`}, `tail_id` = interest entity


## Citation

```bibtex
@inproceedings{xu2027mkge,
  author = {Xu, Ao and Huang, Jinhui and Tang, Wenqiang and Mei, Tiehua and Hou, Hongru and Chen, Hengrui and Liang, Jiaqing and Yang, Deqing},
  title = {Grounding User Preferences in Shared Multimodal Concepts for Recommendation},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  year = {2027},
  doi = {10.1145/3862322.3863599}
}
```
