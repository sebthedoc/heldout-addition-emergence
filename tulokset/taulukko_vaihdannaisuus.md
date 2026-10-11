## Peiliosuus vs. lopullinen pois jätettyjen tarkkuus (40 k askelta, `v2_random_*`)

| osuus | siemen | peilillisiä pois jätetyistä | pois jätetyt oikein | yleistyi |
|---|---|---|---|---|
| 0.075 | 0 | 0.075 | 0.068 | ei |
| 0.075 | 1 | 0.075 | 0.075 | ei |
| 0.075 | 2 | 0.073 | 0.070 | ei |
| 0.1 | 0 | 0.099 | 0.107 | ei |
| 0.1 | 1 | 0.100 | 0.105 | ei |
| 0.1 | 2 | 0.098 | 0.092 | ei |
| 0.12 | 0 | 0.121 | 0.130 | ei |
| 0.12 | 1 | 0.118 | 0.146 | ei |
| 0.12 | 2 | 0.117 | 0.126 | ei |
| 0.13 | 0 | 0.131 | 0.261 | ei |
| 0.13 | 1 | 0.130 | 0.986 | kyllä |
| 0.13 | 2 | 0.127 | 0.349 | ei |
| 0.14 | 0 | 0.140 | 0.149 | ei |
| 0.14 | 1 | 0.139 | 0.986 | kyllä |
| 0.14 | 2 | 0.137 | 0.147 | ei |
| 0.15 | 0 | 0.150 | 0.993 | kyllä |
| 0.15 | 1 | 0.149 | 0.987 | kyllä |
| 0.15 | 2 | 0.148 | 0.991 | kyllä |
| 0.2 | 0 | 0.193 | 0.993 | kyllä |
| 0.2 | 1 | 0.198 | 0.997 | kyllä |
| 0.2 | 2 | 0.200 | 0.996 | kyllä |

## Muistavat mallit pareittain (`v5_malli_f0.1_s*.pt`)

| malli | pois jätetyt | peilillisiä | oikein peilillisistä | oikein peilittömistä | oikeista peilillisiä |
|---|---|---|---|---|---|
| v5_malli_f0.1_s0 | 9000 | 894 | 0.999 | 0.010 | 893/977 |
| v5_malli_f0.1_s1 | 9000 | 902 | 0.958 | 0.007 | 864/918 |
