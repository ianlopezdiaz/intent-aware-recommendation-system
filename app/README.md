# App

Entry point for running the finished pipeline outside a notebook: `cli.py`, the CLI described in the challenge spec's Part 6.

```
python app/cli.py --category "{'title': '...', 'concatenated_tags': '...', 'price': ..., 'minimum_quantity': ..., 'weight': ...}"
python app/cli.py --intent "<user search query>"
python app/cli.py --recommendation "<user search query>"
```

Depends on the trained artifacts in `models/` (produced by notebooks 02, 03, and 05); run those notebooks first if the `.joblib` files are missing.
