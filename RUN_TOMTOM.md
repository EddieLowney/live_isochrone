# First live TomTom run

## 0. Key

```bash
export TOMTOM_API_KEY='...'          # never commit; the config reads key_env
```

The key is never written to disk by this project. `config/tomtom.yaml` stores
only the environment variable *name*.

## 1. Preflight — 0 transactions

```bash
python3 -m isogrid.cli --config config/tomtom.yaml preflight --dry-run
```

Prints the exact request body and the budget arithmetic without sending
anything. Expected: `planned spend 15/cycle -> 2160/day [fits]`.

## 2. Preflight — 2 transactions

```bash
python3 -m isogrid.cli --config config/tomtom.yaml preflight
```

Expected: two cells with a travel time and a free-flow value, then
`free-flow returned  True`.

If it fails:
- **403** — the key lacks Matrix Routing v2 entitlement, or the daily quota is
  gone. The screenshot shows the entitlement, so suspect quota.
- **400** — the request was rejected. The body printed above it is what was
  sent; paste both back for diagnosis.
- **no key** — the export did not take.

## 3. Seed the baseline — 37 transactions

```bash
python3 -m isogrid.cli --config config/tomtom.yaml seed
```

One live sweep. TomTom returns `trafficDelayInSeconds`, so free-flow is derived
per cell rather than measured separately. Expected:
`baseline seeded from 37 live cells`.

## 4. One cycle — 15 transactions

```bash
python3 -m isogrid.cli --config config/tomtom.yaml run --cycles 1
```

Check `cells_measured: 15` and `cells_failed`. Any failures are cells in the
harbour; three consecutive failures quarantines them.

## 5. Serve

```bash
python3 -m isogrid.cli --config config/tomtom.yaml serve
# http://127.0.0.1:8000
```

Scheduler runs a cycle every 600s.

## Day-one budget

| Step | Transactions |
|---|---|
| preflight | 2 |
| seed | 37 |
| 144 cycles x 15 | 2,160 |
| **total** | **2,199** of 2,250 usable |

Fits, with ~50 spare. From day two the seed cost disappears.

## Watch these

- `cells_failed` in the first few cycles — water cells. `isogrid.cli mask`
  shows what got quarantined once they trip the threshold.
- `holdout_mae_s` — will look bad for the first ~20 cycles (3.5 hours) while
  the factor model has no history and falls back to IDW.
- `budget_remaining` — should land near zero at midnight UTC, not at noon.

## Rotate the key

The dashboard shows `Rotated: Never`. Rotate once testing is done, then
re-export. Nothing in the project needs updating.
