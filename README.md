# Code Python — article « Optimal Periodic Impulsive Insulin Delivery for Type 1 Diabetes »

## Contenu
| Fichier | Rôle |
|---|---|
| `bergman_impulsive.py` | Bibliothèque : modèle (G, X, I, m), intégration avec sauts, coût, gradient adjoint exact, gradient par sensibilités directes, optimisation, orbite périodique, gradient périodique, matrice de monodromie (Floquet), indices TIR/TAR/TBR, population virtuelle |
| `run_all.py` | Reproduit **tous** les nombres des tableaux, **affiche** les 4 figures et les enregistre (EPS + PDF) ; écrit `results.txt` |
| `results.txt` | Journal de la dernière exécution |

## Installation et exécution
```
python3 -m pip install -r requirements.txt   # numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2
python3 run_all.py                  # calcul complet (4 à 5 min) : tableaux dans results.txt,
                                    # puis AFFICHAGE des 4 figures de l'article et enregistrement (EPS + PDF)
python3 run_all.py --figures-only   # seulement les 4 figures, affichées (1 à 2 min)
python3 run_all.py --no-show        # sans fenêtre (serveur) : enregistre seulement
python3 run_all.py --outdir DOSSIER # change le dossier de sortie (défaut : ./figures)
```
Les figures s'affichent à la fin du calcul ; fermer les fenêtres termine le programme. Sans écran graphique, le programme l'indique et enregistre simplement les figures. Les figures obtenues sont identiques pixel pour pixel à celles de l'article (vérifié). Testé avec Python 3.13.

## Utilisation de la bibliothèque
```python
from bergman_impulsive import *
r = optimise([7, 10, 5, 0])                 # doses optimales (SLSQP, gradient exact)
kappa = r.x                                 # -> [7.1074, 10.718, 5.2903, 0.]
print(cost(kappa), day_indices(kappa))      # J et TIR/TAR/TBR
g, _ = gradient_adjoint(kappa)              # gradient adjoint rho*kappa_j + lambda_3(t_j)
sc = NOMINAL.with_phys(p3=4 * PHYS["p3"])   # patient avec p3 multiplié par 4
print(day_indices(kappa, sc))
```
Pour changer un repas, un horaire d'injection ou un poids du coût : `Scenario(carbs=(...), meal_t=(...), inj_t=(...), w={...})`.

## Remarque sur les figures EPS
Le format EPS ne gère pas la transparence : les zones colorées utilisent des couleurs pleines. Le texte est incorporé en Type 42.

## Reproductibilité
- Python 3.13 ; versions exactes dans `requirements.txt`.
- Toutes les graines aléatoires sont fixées (`population(..., seed=0)` pour les figures, graines 0 à 5 pour le contrôle de stabilité).
- Aucune donnée externe : tous les paramètres sont dans `bergman_impulsive.py` (dictionnaires `PHYS`, `WEIGHTS`, `NOMINAL`).
- `results.txt` contient la sortie de référence ; une réexécution doit redonner les mêmes nombres (les durées affichées peuvent varier).
- Licence : MIT (`LICENSE`). Citation : `CITATION.cff`.
