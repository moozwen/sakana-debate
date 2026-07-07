```bash
(sakana-debate) nakimura@Mac sakana-debate % uv run python analysis/analyze.py
=== accuracy per condition (mean ± 95%CI over seeds) ===
N=1 R=0: 0.943 ± 0.014  (seeds=[1, 2, 3])
N=1 R=2: 0.947 ± 0.026  (seeds=[1, 2, 3])
N=2 R=2: 0.950 ± 0.012  (seeds=[1, 2, 3])
N=3 R=0: 0.948 ± 0.007  (seeds=[1, 2, 3])
N=3 R=1: 0.958 ± 0.019  (seeds=[1, 2, 3])
N=3 R=2: 0.950 ± 0.000  (seeds=[1, 2, 3])
N=3 R=3: 0.953 ± 0.007  (seeds=[1, 2, 3])
N=4 R=2: 0.958 ± 0.019  (seeds=[1, 2, 3])
N=6 R=2: 0.955 ± 0.012  (seeds=[1, 2, 3])
wrote analysis/results/fig_r_curve.png
wrote analysis/results/fig_n_curve.png

=== RQ5: unanimity vs accuracy（最終ラウンドの全会一致で層別） ===
      cond  coverage  acc(unanimous)  acc(split)
N=2 R= 2    98.7%           0.959       0.250
N=3 R= 0    91.7%           0.975       0.660
N=3 R= 1    96.8%           0.969       0.632
N=3 R= 2    98.7%           0.959       0.250
N=3 R= 3    99.0%           0.960       0.333
N=4 R= 2    97.5%           0.974       0.333
N=6 R= 2    97.3%           0.967       0.500
→ acc(unanimous) ≫ acc(split) なら「全会一致でなければ人間へ」が有効な自己申告（coverage がそのときの自動処理率）
```
