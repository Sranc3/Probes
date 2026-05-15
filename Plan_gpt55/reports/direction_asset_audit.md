# Direction Asset Audit

## Metadata

- Run dir: `/zhutingqi/song/layer_level/output/run_20260403_050400`
- Train question count: `139`
- Semantic threshold: `0.665403`
- Gate threshold: `0.100681`

## Deploy Selection Split Overlap

- Selected count: `200`
- Train overlap: `139`
- Val overlap: `30`
- Test overlap: `31`

## Site Summaries

| Site | Pos/Neg | Raw norm | Median step norm | Projection separation | Cohen d |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp.output|layer_18` | `46/93` | `4.7022` | `29.8871` | `4.7022` | `1.9191` |
| `mlp.output|layer_24` | `46/93` | `23.9535` | `102.5506` | `23.9535` | `1.4620` |

## Interpretation

- Strong projection separation supports using the site as a diagnostic probe.
- Weak or outlier-dominated separation argues against fixed-vector deployment control.
- If deploy selection overlaps train heavily, deployment-style claims should be separated from held-out claims.
