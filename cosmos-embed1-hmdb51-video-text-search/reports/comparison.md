# Cosmos-Embed1 HMDB51 video-text search — baseline vs finetuned

Test set: 306 clips, 51 classes (identical for both phases).

| metric | baseline (zero-shot) | finetuned (LoRA) | delta |
|---|---:|---:|---:|
| accuracy | 0.6569 | 0.7745 | +0.1176 |
| precision_macro | 0.6482 | 0.7961 | +0.1478 |
| recall_macro | 0.6569 | 0.7745 | +0.1176 |
| f1_macro | 0.6240 | 0.7641 | +0.1400 |
| precision_weighted | 0.6482 | 0.7961 | +0.1478 |
| recall_weighted | 0.6569 | 0.7745 | +0.1176 |
| f1_weighted | 0.6240 | 0.7641 | +0.1400 |
| v2t_recall@1 | 0.6569 | 0.7745 | +0.1176 |
| v2t_recall@3 | 0.8301 | 0.9281 | +0.0980 |
| v2t_recall@5 | 0.8922 | 0.9510 | +0.0588 |
| v2t_recall@10 | 0.9477 | 0.9771 | +0.0294 |
| t2v_recall@1 | 0.8235 | 0.9412 | +0.1176 |
| t2v_recall@5 | 0.9216 | 0.9804 | +0.0588 |
| t2v_recall@10 | 0.9608 | 1.0000 | +0.0392 |
| t2v_mAP | 0.7011 | 0.8269 | +0.1258 |
| t2v_median_rank_first_relevant | 1.0000 | 1.0000 | +0.0000 |

## Container evaluate top-k (baseline)

| container metric | value |
|---|---:|
| top1 | 0.6601 |
| top10 | 0.9444 |
| top3 | 0.8268 |
| top5 | 0.8922 |

cross-check top-1: custom 0.6569 vs container 0.6601

## Container evaluate top-k (finetuned)

| container metric | value |
|---|---:|
| top1 | 0.7778 |
| top10 | 0.9771 |
| top3 | 0.9281 |
| top5 | 0.9510 |

cross-check top-1: custom 0.7745 vs container 0.7778

## Notes

- v2t_recall@k equals top-k classification accuracy (one relevant class prompt per video).
- t2v metrics are multi-positive (6 relevant clips per class query); the container's
  caption-level retrieval R@K is ill-defined with duplicate captions and is not reported.
- Full similarity matrices and per-video top-10 score logs live next to each phase's
  metrics.json (sim_video_x_class.csv, similarity_scores.npz, retrieval_log.csv).
