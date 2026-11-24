# MKGE Prompt Templates

This directory contains the prompt templates used by MKGE for item-side concept
induction and temporal preference grounding. They are plain-text transcriptions
of the templates reported in the original manuscript appendix; LaTeX-only
formatting has been removed without changing the prompt instructions.

In the current paper, the terms **entity** and **attribute** used in these
templates correspond to a **canonical multimodal concept**. The item-side caller
supplies the item image and short title as multimodal context. Bracketed examples
and ellipses in the templates indicate runtime content to be populated from the
current item, induced clusters, canonical vocabulary, or user history.

## Runtime payloads

- `01_discover.txt` and `06_constrain.txt` are sent with the item image and its
  short title. For Constrain, the complete dataset-specific vocabulary replaces
  the abbreviated example block.
- The current relation clusters are appended to `02_standardize_relations.txt`.
- The current relation-specific entity clusters are appended to Files 03--05.
- The relation-grouped, history-supported candidate pool and evidence counts
  replace the examples in `07_ground_short_term.txt`. When at least ten
  candidates exist, the model selects exactly ten; otherwise all available
  supported candidates are retained rather than inventing new concepts.
- The four recent short-term states and previous long-term state replace the
  examples in `08_ground_long_term.txt`.

## Files

1. `01_discover.txt`: open-ended relation--entity discovery.
2. `02_standardize_relations.txt`: relation-cluster refinement.
3. `03_standardize_entities_split.txt`: splitting semantically conflicting entity clusters.
4. `04_standardize_entities_merge.txt`: merging clusters that express the same preference.
5. `05_standardize_entities_rename.txt`: assigning canonical cluster names.
6. `06_constrain.txt`: catalog-wide extraction under the standardized vocabulary.
7. `07_ground_short_term.txt`: selecting supported concepts from one 21-day bucket.
8. `08_ground_long_term.txt`: updating persistent concepts every four buckets.

The Standardize stage uses Files 02--05. Files 03--05 are applied sequentially
to each relation-specific entity clustering result.
