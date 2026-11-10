# Vérification du top-1 RRF (2026-10-28)

**`rrf.json` et `kernel/input/candidates.jsonl` : aucun écart.** Sur les 129 tâches, les top-5 sont identiques. Les deux viennent de `evaluate.base_ranking` (RRF complète). Les 10 candidats de chaque ligne e2b/e4b (conditions trace et notrace) sont exactement le top-10 exporté.

**Recalcul à partir de `results/separes.jsonl` : 1 tâche diffère, fastapi_15023.**
- Top-1 stocké : `fastapi/routing.py` (0,04641).
- Top-1 recalculé : `scripts/docs.py` (0,04528).
- Cause : une troncature, pas une égalité ni une normalisation de chemins. `triggers.export` ne garde que le top-10 de chaque classement séparé. Or `routing.py` est 12e en BM25 fichiers (2e en nœuds, 1er en graphe). Sa part 1/72 disparaît donc du recalcul.
- L'or (`docs_src/stream_data/tutorial002_py310.py`) n'est aucun des deux fichiers.

**Effet sur le writeup : nul.**
- Acc@1 RRF = 43/129 = 0,333, inchangé.
- Toutes les règles de `triggers.py` prennent `t["base"]` (le classement stocké) pour la marge, le score du 1er et le top-1 de « RRF soutenu ». `separes.jsonl` ne sert qu'aux top-1 séparés, que la troncature ne touche pas.
- Pour fastapi_15023, la règle « RRF soutenu 2/3 » se déclenche dans les deux cas (soutien 1 avec le top-1 stocké, 0 avec le recalculé).
- La phrase du § 118 (« une tâche ») est exacte. Le chiffre « 1 à 3 » ne se vérifie pas.

**Égalités exactes entre les rangs 1 et 2 : 2 tâches.** `_sorted` les départage déjà de façon stable, par chemin croissant.
- fastapi_9753 : sans effet, l'or n'est aucun des deux fichiers.
- requests_6592 : l'or `status_codes.py` est classé 2e derrière `sessions.py`. Avec l'autre ordre, Acc@1 RRF vaudrait 44/129 = 0,341. C'est une sensibilité, pas un écart.

**Correctif : aucun.** Le tri est déjà déterministe, et la troncature n'affecte aucun chiffre publié. Le code n'a pas été modifié et les tests n'ont pas été relancés.
