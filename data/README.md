# Data

Raw connectomes are intentionally excluded from Git.

Use `scripts/download_malecns.py` for official MaleCNS v1.0 bulk files or `scripts/neuprint_query.py` for a small authenticated neuPrint query.

Expected local layout:

```text
data/
  raw/
    malecns-v1.0/
      body-annotations-....feather
      connectome-weights-....feather
```

Before committing any derived data, document:

- upstream URL/dataset version,
- extraction code and parameters,
- date obtained,
- upstream license,
- whether the derived artifact can legally be redistributed.
