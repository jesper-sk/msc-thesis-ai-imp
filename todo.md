# To do - 2023-06-19

## Conceptors

**Framework test**
- 2D dimensionality reduction? UMAP --> conceptor?
- Assertions (is positive semidefinite, represents ellipsoid, etc)
- Ellipse plot fix (which is it?)
  - Contact paul bricman?
- Distance and subsumption metrics

**Subsumption**
- Research different heuristics for lowner orderedness 
  - Lowner: X is more abstract than Y iff X-Y is positive semidefinite (i.e., e.g. the eigenvalues of X-Y are nonnegative)
  - Option 1: (Bricman) Use sum of eigenvalues of difference matrix
    - Implies that one negative eigenvalue can be compensated for by another (large) positive eigenvalue
  - Option 2: Amount of nonnegative eigenvalues divided by total
    - Or: Where in the graph of ordered eigenvalues does one intersect zero
  - Option 3: 
- "Dashboard"/ipynb for comparing heuristics/plotting eigenvalues etc? 
  - Be able to "play" with two ellipses, and see how the changes impact the heuristics
    - Convert ellipse to matrix representation
- Make plot: UMAP of a word's different senses, together with the different conceptors + their overlap

## Clustering

**EWISER for CoarseWSD-20**
- Pre-trained ewiser uses WordNet vocab which does not include the correct senses for cwsd, so we need to retrain ewiser for this.
  - Is this worth the time? Herbert probably thinks not


## Dataset(s)
**Is CoarseWSD-20 suitable for abstraction ordering?**
- Only 20 words which are not inherently ordered in abstraction
  - Alternative would be to try to cluster them hierarchically with distance metric based on lowner orderedness, and then see whether the combined ($A\vee B$) conceptor actually subsumes A and B 
- Alternative: use (words from) (one of the) datasets from testing framework (Senseval et al)? 
  - Some work to format data for 'naive' BERT vectorisation + k-means
  - Does mean EWISER works ootb
  - Probably contains more words from different hierarchies, makes qualititave research easier (i.e. comparison to intuition)

# Meeting 2023 07 13

* Showing off conceptor ellipses and ellipsoids
* Different kinds of heuristics
  * Strict loewner
  * Bricmans weighted eigensum
  * Ratio positive ~ negative eigenvalues
  * Positive->negative cutoff point (divided by N)
* Observation: Eigenvalues of difference matrix kinda similar to difference of eigenvalues (ordered)
* Observation: If for C1 and C2, we have L1, L2 < N, then there are (N - L1) and (N - L2) zero singular values (logical, amnt of nonzero singvals equal to matrix rank)
* Observation: Given above, if we take (C1 - C2), and we also have L1 + L2 < N, we have that there are L1 positive eigenvalues and L2 negative eigenvalues in the difference matrix. 
  * Implication: We probably want to ignore eigenvalues close to zero (floating-point error), as the size of L does not correlate with abstraction.
* Desmos for aperture values https://www.desmos.com/calculator/85uh7ezwg0
* Singular values of correlation matrix