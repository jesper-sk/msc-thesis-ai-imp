# %% Start
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from matplotlib.patches import Ellipse as MEllipse
from sklearn.decomposition import PCA

from wsd.conceptors.conceptor import *
from wsd.util.angle import rad_to_deg

warnings.filterwarnings("ignore")

path = Path("../out/split/bert-base-uncased")
senses = list(path.glob("*.npy"))


def embeddings(stem: str):
    return np.load([sense for sense in senses if sense.stem.strip() == stem][0])


conceptor = Conceptor.from_state_matrix


def plot_ellipses(ell: Ellipse | Conceptor, *ells: Ellipse | Conceptor):
    fig, ax = plt.subplots()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    colors = ["red", "blue", "green", "purple", "yellow", "orange"]

    for idx, item in enumerate([ell] + list(ells)):
        ellipse = item if isinstance(item, Ellipse) else item.ellipse()

        p = ax.add_patch(
            MEllipse(
                (0, 0),
                ellipse.semiaxis_x,
                ellipse.semiaxis_y,
                rad_to_deg(ellipse.angle),
            )
        )
        p.set_edgecolor(colors[idx])
        p.set_color(colors[idx])
        p.set_alpha(0.6)

    plt.show()


def plot_ellipsoids(ell: np.ndarray | Conceptor, *ells: np.ndarray | Conceptor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["red", "blue", "green", "purple", "yellow", "orange"]

    all_ellipsoids = [ell] + list(ells)
    all_ellipsoids = [
        e.ellipsoid() if hasattr(e, "ellipsoid") and len(e.shape) == 2 else e
        for e in all_ellipsoids
    ]

    for idx, item in enumerate(all_ellipsoids):
        ax.plot_wireframe(*item, rstride=4, cstride=4, color=colors[idx], alpha=0.6)

    lim = np.stack([e.max(axis=(1, 2)) for e in all_ellipsoids]).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    plt.show()


def plot_eigs(c: Conceptor):
    eigs = np.linalg.eigvalsh(c)
    plt.plot(sorted(eigs))
    # plt.yscale('log')


def weighted_eigsum(diff: np.ndarray):
    eigs = np.linalg.eigvalsh(diff)
    return eigs.sum() / len(eigs)


def posneg_fraction(diff: Conceptor, tol=1e-3, do_print=True):
    eigs = np.sort(np.linalg.eigvals(diff))
    where_neg = np.argwhere(eigs < -tol)
    where_pos = np.argwhere(eigs > tol)

    neg_amount = where_neg[-1] if len(where_neg) > 0 else 0
    pos_cutoff = where_pos[0] if len(where_pos) > 0 else len(eigs)
    pos_amount = len(eigs) - pos_cutoff

    if do_print:
        print(
            f"pos: {pos_amount}, neg: {neg_amount}, zero: {len(eigs) - (pos_amount + neg_amount)}"
        )

    return ((2 * pos_amount) / (neg_amount + pos_amount)) - 1


def cutoff(diff: Conceptor, tol=1e-3):
    eigs = np.sort(np.linalg.eigvals(diff))
    where_neg = np.argwhere(eigs < -tol)
    where_pos = np.argwhere(eigs > tol)

    neg_amount = where_neg[-1] if len(where_neg) > 0 else 0
    pos_cutoff = where_pos[0] if len(where_pos) > 0 else len(eigs)

    return (((neg_amount + pos_cutoff) / len(eigs)) - 1) * -1


def nonzero_mean(diff: Conceptor, tol=1e-3):
    eigs = np.sort(np.linalg.eigvals(diff))
    where_neg = np.argwhere(eigs < -tol)
    where_pos = np.argwhere(eigs > tol)

    pos_mean = eigs[where_pos].mean() if len(where_pos) > 0 else 0
    neg_mean = eigs[where_neg].mean() if len(where_neg) > 0 else 0

    return ((2 * pos_mean) / (pos_mean - neg_mean)) - 1


eigvals = np.linalg.eigvals
sorted_eigvals = lambda x: sorted(np.linalg.eigvals(x))


def sigvals(x: np.ndarray):
    return np.linalg.svd(x)[1]


def plot_corr_sigvals(embs):
    corr_music = (embs.T @ embs) / len(embs)
    plt.plot(sigvals(corr_music))
    plt.yscale("log")


def loewner(diff: Conceptor, tol: float = 1e-3) -> int:
    """Checks whether the given two conceptors are loewner-ordered.

    Parameters
    ----------
    a : Conceptor
        The first conceptor
    b : Conceptor
        The second conceptor
    tol : float, optional
        The floating-point comparison tolerance, by default 1e-8

    Returns
    -------
    int
        1 iff a >= b; -1 iff a <= b; 0 if no loewner ordering is present between a and b
    """
    diff_eigvals = la.eigvals(diff)
    if np.all(diff_eigvals + tol >= 0):
        return 1
    if np.all(diff_eigvals - tol <= 0):
        return -1
    return 0


# %% Load
print("Embeddings...")
emb_music = embeddings("bow_bow_(music)")
emb_ship = embeddings("bow_bow_(ship)")
emb_arrow = embeddings("bow_bow_and_arrow")
emb_ap_inc = embeddings("apple_apple")
emb_ap_frt = embeddings("apple_apple_inc.")
emb_all = np.concatenate((emb_music, emb_ship, emb_arrow))


def emb_class(idx):
    if idx < len(emb_music):
        return "music"
    if idx < len(emb_music) + len(emb_ship):
        return "ship"
    return "arrow"


emb_classes = [emb_class(idx) for idx in range(len(emb_all))]

aperture = 0.3

print("Conceptors...")
co_music = conceptor(emb_music, aperture)
co_ship = conceptor(emb_ship, aperture)
co_arrow = conceptor(emb_arrow, aperture)
co_ap_inc = conceptor(emb_ap_inc, aperture)
co_ap_frt = conceptor(emb_ap_frt, aperture)

co2_music = conceptor(emb_music[:, :2], 0.07)
co2_ship = conceptor(emb_ship[:, :2], 0.07)
co2_arrow = conceptor(emb_arrow[:, :2], 0.07)
co3_music = conceptor(emb_music[:, :3], 0.07)
co3_ship = conceptor(emb_ship[:, :3], 0.07)
co3_arrow = conceptor(emb_arrow[:, :3], 0.07)

# %% UMAP scatter
import umap

colors = {
    "music": "red",
    "ship": "blue",
    "arrow": "green",
}
emb2d_all = umap.UMAP().fit_transform(emb_all)
plt.scatter(emb2d_all[:, 0], emb2d_all[:, 1], c=[colors[cl] for cl in emb_classes])

# %% Ellipses
plot_ellipses(co2_music, co2_ship, co2_arrow)

# %% Ellipsoids
plot_ellipsoids(co3_music, co3_ship, co3_arrow)

# %% Aperture estimation
# I picked aperture in such a way that the asymptote of f(x) = x / (x + a**-2) is around
# the max signular value of the embedding correlation matrix (around 3000).
# Via https://www.desmos.com/calculator/85uh7ezwg0, this is 0.3.
# Todo: Read up and look at gradient of squared Frobenius norm
plot_corr_sigvals(emb_music)
plot_corr_sigvals(emb_ship)
plot_corr_sigvals(emb_arrow)

# %%
plt.plot(sigvals(co_music))
plt.yscale("log")
# %%
plt.plot(sorted_eigvals(co_music - co_ship))
print(f"len(emb_music): {len(emb_music)}; len(emb_ship): {len(emb_ship)}")
posneg_fraction(co_music - co_ship)


# %%
def heurs(c1, c2):
    print(f"Loewner: {loewner(c1-c2)}")
    print(f"weighted eigsum: {weighted_eigsum(c1-c2):.2f}")
    print(f"posneg_ratio: {posneg_fraction(c1-c2, do_print=False)[0]:.2f}")
    print(f"cutoff: {cutoff(c1-c2)[0]:.2f}")
    print(f"posneg_mean: {nonzero_mean(c1-c2):.2f}")
    print(f"pncf: {posneg_count_fraction(c1, c2):.2f}")
    print(f"pnmf: {posneg_magnitude_fraction(c1, c2):.2f}")


# %%
print("co_music - co_ship (red)")
heurs(co_music, co_ship)
x = sorted_eigvals(co_music - co_ship)
plt.plot(x, color="red")
plt.hlines(0, 0, len(x), linestyles="--")

# %%
print("co_music - co_arrow (green)")
heurs(co_music, co_arrow)
x = sorted_eigvals(co_music - co_arrow)
plt.plot(x, color="red")
plt.hlines(0, 0, len(x), linestyles="--")

# %%
print("co_ship - co_arrow (blue)")
heurs(co_ship, co_arrow)
x = sorted_eigvals(co_ship - co_arrow)
plt.plot(x, color="red")
plt.hlines(0, 0, len(x), linestyles="--")

# %%
print("co_ship - conj(co_music, co_arrow) (purple)")
conj = co_music.conj(co_arrow)
heurs(co_ship, conj)
x = sorted_eigvals(co_ship - conj)
plt.plot(x, color="red")
plt.hlines(0, 0, len(x), linestyles="--")

# %%
print("co_music - disj(co_ship, co_arrow) (yellow)")
disj = co_ship.disj(co_arrow)
heurs(co_music, disj)
x = sorted_eigvals(co_music - disj)
plt.plot(x, color="red")
plt.hlines(0, 0, len(x), linestyles="--")
# %%
