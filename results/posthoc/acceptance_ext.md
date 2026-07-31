## MPS 2x2x2 pattern histograms (F.1)

| environment | JSD (bits) | band | verdict |
|---|---|---|---|
| lobe | 0.00013 | 0.00030 | inside |
| channel:PV_SHOESTRING | 0.00041 | 0.00015 | outside |
| channel:CB_LABYRINTH | 0.00015 | 0.00018 | inside |
| channel:CB_JIGSAW | 0.00040 | 0.00034 | near |
| channel:SH_DISTAL | 0.00025 | 0.00008 | outside |
| channel:SH_PROXIMAL | 0.00023 | 0.00018 | near |
| channel:MEANDER_OXBOW | 0.00011 | 0.00012 | inside |
| delta | 0.00015 | 0.00023 | inside |

## Sand run-length statistics (F.2)

| environment | axis | median ref→gen | p90 ref→gen | W1 (band) | verdict |
|---|---|---|---|---|---|
| lobe | z | 4→4 | 10→10 | 0.141 (0.161) | inside |
| lobe | x | 9→9 | 27→26 | 0.179 (0.234) | inside |
| channel:PV_SHOESTRING | z | 4→4 | 8→8 | 0.157 (0.134) | near |
| channel:PV_SHOESTRING | x | 6→6 | 19→18 | 0.346 (0.185) | near |
| channel:CB_LABYRINTH | z | 4→4 | 10→10 | 0.166 (0.117) | near |
| channel:CB_LABYRINTH | x | 8→8 | 30→29 | 0.297 (0.170) | near |
| channel:CB_JIGSAW | z | 3→3 | 8→8 | 0.231 (0.092) | outside |
| channel:CB_JIGSAW | x | 6→5 | 27→26 | 0.661 (0.219) | outside |
| channel:SH_DISTAL | z | 5→4 | 16→16 | 0.643 (0.097) | outside |
| channel:SH_DISTAL | x | 21→22 | 55→55 | 0.590 (0.508) | near |
| channel:SH_PROXIMAL | z | 4→4 | 11→10 | 0.321 (0.079) | outside |
| channel:SH_PROXIMAL | x | 11→10 | 43→44 | 0.684 (0.273) | outside |
| channel:MEANDER_OXBOW | z | 5→5 | 13→13 | 0.306 (0.232) | near |
| channel:MEANDER_OXBOW | x | 12→11 | 34→34 | 0.239 (0.085) | outside |
| delta | z | 5→4 | 11→10 | 0.238 (0.104) | outside |
| delta | x | 9→8 | 41→40 | 0.513 (0.770) | inside |

## Artifact rate: bodies < 8 voxels (F.5)

| environment | mean/vol ref | gen | band | verdict |
|---|---|---|---|---|
| lobe | 49.79 | 49.77 | 0.59 | inside |
| channel:PV_SHOESTRING | 50.62 | 56.38 | 0.71 | outside |
| channel:CB_LABYRINTH | 77.68 | 67.38 | 7.46 | near |
| channel:CB_JIGSAW | 169.62 | 160.71 | 17.95 | inside |
| channel:SH_DISTAL | 1.34 | 3.69 | 0.10 | outside |
| channel:SH_PROXIMAL | 35.88 | 37.72 | 1.50 | near |
| channel:MEANDER_OXBOW | 90.45 | 85.06 | 0.72 | outside |
| delta | 39.38 | 43.79 | 1.80 | outside |

### Ensemble (b): artifacts vs well count

| config | n vols | mean artifacts/vol | frac ≥1 artifact [Wilson 95%] |
|---|---|---|---|
| 1well | 688 | 67.81 | 0.974 [0.959, 0.983] |
| 2wells | 680 | 60.78 | 0.981 [0.968, 0.989] |
| 3wells | 680 | 59.24 | 0.981 [0.968, 0.989] |
