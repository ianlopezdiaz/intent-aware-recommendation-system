# Data

## Layout

- `raw/` — original, immutable data. Contains `elo7_recruitment_dataset.csv`, the dataset provided for the challenge. Never edit in place.
- `processed/` — final, model-ready datasets and cached artifacts produced by the notebooks/`src/data` pipeline.

## `raw/elo7_recruitment_dataset.csv`

38,507 rows × 15 columns. **One row = one product click resulting from a search query** — the same `product_id` can (and does) appear multiple times, once per click event, so `product_id` is not a unique key at the row level (29,801 unique product IDs across 38,507 rows).

| Column | Dtype | Nulls | Notes |
|---|---|---|---|
| `product_id` | int | 0 | Not unique per row (see above). |
| `seller_id` | int | 0 | |
| `query` | string | 0 | Raw search query text, Portuguese. |
| `search_page` | int | 0 | Range 1–5 per the challenge spec. |
| `position` | int | 0 | Position of the product on the results page, range 0–38 per the challenge spec. |
| `title` | string | 0 | Seller-provided product title, Portuguese. |
| `concatenated_tags` | string | 2 | Seller-provided tags, space-separated. |
| `creation_date` | datetime | 0 | Product creation timestamp. Range observed: 2008-11-05 to 2020-01-03. |
| `price` | float | 0 | BRL. Observed range 0.07–11,509.38, heavily right-skewed (mean 84.05, median 28.49). Seller-reported — the challenge spec explicitly warns these "may not always reflect realistic market values." |
| `weight` | float | 58 | Grams. Observed range 0–65,009, also heavily right-skewed (mean 361.8, median 9.0). |
| `express_delivery` | int (0/1) | 0 | Boolean flag. ~78% of rows are 1. |
| `minimum_quantity` | int | 0 | Minimum purchase quantity. |
| `view_counts` | int | 0 | Views in the prior 3 months. |
| `order_counts` | float | 20,390 (~53%) | Purchases in the prior 3 months. **Null almost certainly means zero orders in the window, not missing data** — treat as `0` after confirming, don't drop or impute as unknown. |
| `category` | string | 0 | Target label for Part 2 (classification). Six values, imbalanced: `Lembrancinhas` 17,759, `Decoração` 8,846, `Bebê` 7,026, `Papel e Cia` 2,777, `Outros` 1,148, `Bijuterias e Jóias` 951. |

No ground-truth label exists for search intent (Part 3) — that has to be constructed, see `../notebooks/03_search_intent_modeling.ipynb` for the methodology.

Full column semantics and the original problem statement: `../docs/elo7-ds-challenge-en.md`.
