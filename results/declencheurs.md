# Déclencheurs du reclassement (129 tâches)

Frontière de Pareto Acc@1 contre secondes par tâche (moyenne sur toutes les tâches, coût nul sans déclenchement ; T4). IC bootstrap 95 %. Seuils choisis sur ces mêmes tâches : optimiste. Toutes les lignes : `declencheurs.json`.

| Règle | Modèle | Décl. | Acc@1 [IC] | Acc@5 | MRR | s/tâche | jetons/tâche |
|---|---|---:|---|---:|---:|---:|---:|
| jamais | e2b | 0 | 0.333 [0.26 ; 0.42] | 0.659 | 0.493 | 0.00 | 0 |
| marge_abs<0.0001 | e2b | 5 | 0.341 [0.26 ; 0.43] | 0.659 | 0.497 | 0.28 | 122 |
| marge_abs<0.00025 | e2b | 8 | 0.349 [0.26 ; 0.43] | 0.659 | 0.504 | 0.37 | 190 |
| accord_2sur3 | e2b | 11 | 0.372 [0.29 ; 0.46] | 0.682 | 0.524 | 0.47 | 241 |
| rrf_soutenu_2sur3 | e2b | 23 | 0.395 [0.31 ; 0.48] | 0.698 | 0.545 | 0.99 | 537 |
| rrf_soutenu_2sur3 | e4b | 23 | 0.403 [0.32 ; 0.49] | 0.705 | 0.551 | 1.33 | 536 |
| marge_abs<0.001 | e2b | 60 | 0.419 [0.33 ; 0.50] | 0.713 | 0.560 | 2.59 | 1422 |
| accord_3sur3 | e2b | 79 | 0.496 [0.41 ; 0.59] | 0.713 | 0.607 | 3.10 | 1837 |
| marge_abs<0.002 | e2b | 107 | 0.504 [0.42 ; 0.59] | 0.744 | 0.616 | 4.32 | 2621 |
| accord_3sur3 | e4b | 79 | 0.535 [0.45 ; 0.63] | 0.752 | 0.638 | 4.85 | 1839 |
| marge_abs<0.004 | e2b | 126 | 0.543 [0.46 ; 0.63] | 0.752 | 0.646 | 4.94 | 3040 |
| toujours | e2b | 129 | 0.550 [0.47 ; 0.64] | 0.752 | 0.650 | 5.02 | 3106 |
| marge_abs<0.002 | e4b | 107 | 0.574 [0.49 ; 0.66] | 0.791 | 0.670 | 6.19 | 2620 |
| marge_abs<0.004 | e4b | 126 | 0.612 [0.53 ; 0.70] | 0.806 | 0.700 | 7.31 | 3039 |
| toujours | e4b | 129 | 0.620 [0.53 ; 0.71] | 0.806 | 0.704 | 7.55 | 3107 |

Cascade : e2b sur toutes les tâches, e4b seulement si le top-1 e2b diffère du top-1 RRF ; « Décl. » = appels e4b. Absente de la frontière : dominée par e4b seul, ou par e4b sous marge_abs<0.002, plus rapides à Acc@1 égale ou supérieure.

Classements séparés : BM25 fichiers, BM25 nœuds, graphe à 1 saut (le code n'a pas d'embeddings). accord_3sur3 équivaut à score1<0.049 (3/61 = premier partout).

## Validation par dépôt et tests appariés

Leave-one-repo-out (129 tâches, 4 dépôts) : seuil et modèle choisis sur les trois autres dépôts (Acc@1 maximale à coût ≤ celui de accord_3sur3+e2b sur ces dépôts), appliqués au dépôt tenu à l'écart. Référence sans validation : Acc@1 0.496, 3.10 s/tâche.

| Famille | Acc@1 [IC] | s/tâche | Choix par pli (dépôt : règle, modèle) |
|---|---|---:|---|
| marge_abs | 0.450 [0.36 ; 0.54] | 3.10 | rich : marge_abs<0.0005, e4b ; httpx : marge_abs<0.001, e2b ; fastapi : marge_abs<0.001, e4b ; requests : marge_abs<0.001, e2b |
| score1 | 0.496 [0.41 ; 0.59] | 3.10 | rich : score1<0.049, e2b ; httpx : score1<0.049, e2b ; fastapi : score1<0.049, e2b ; requests : score1<0.049, e2b |
| choix_libre | 0.403 [0.32 ; 0.49] | 2.38 | rich : rrf_soutenu_2sur3, e4b ; httpx : accord_3sur3, e2b ; fastapi : accord_3sur3, e2b ; requests : accord_3sur3, e2b |

Tests appariés (mêmes tâches) : bootstrap des différences d'Acc@1 (10 000 tirages) et McNemar exact (a seul juste / b seul juste).

| a | b | Acc@1 a − b [IC] | p bootstrap | a seul / b seul | p McNemar |
|---|---|---|---:|---:|---:|
| toujours+e4b | jamais+e2b | +0.287 [+0.21 ; +0.37] | < 0.0001 | 38 / 1 | 1.5e-10 |
| toujours+e2b | jamais+e2b | +0.217 [+0.13 ; +0.30] | < 0.0001 | 34 / 6 | 8.4e-06 |
| toujours+e4b | toujours+e2b | +0.070 [+0.02 ; +0.12] | 0.0208 | 12 / 3 | 0.0352 |
| accord_3sur3+e2b | jamais+e2b | +0.163 [+0.09 ; +0.24] | < 0.0001 | 24 / 3 | 4.9e-05 |
| accord_3sur3+e2b | toujours+e2b | -0.054 [-0.11 ; +0.00] | 0.0538 | 3 / 10 | 0.0923 |
| accord_3sur3+e2b | toujours+e4b | -0.124 [-0.19 ; -0.06] | < 0.0001 | 2 / 18 | 4.0e-04 |
| cv_marge_abs | jamais+e2b | +0.116 [+0.06 ; +0.18] | < 0.0001 | 15 / 0 | 6.1e-05 |
| cv_score1 | jamais+e2b | +0.163 [+0.09 ; +0.24] | < 0.0001 | 24 / 3 | 4.9e-05 |
| cv_choix_libre | jamais+e2b | +0.070 [+0.02 ; +0.13] | 0.0210 | 12 / 3 | 0.0352 |
