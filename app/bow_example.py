# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import umap
from matplotlib.patches import Ellipse as MEllipse
from sklearn.decomposition import PCA

from wsd.conceptors.conceptor import Conceptor, Ellipse
from wsd.util.angle import rad_to_deg

warnings.filterwarnings("ignore")

path = Path("../out/split/bert-base-uncased")
senses = list(path.glob("*.npy"))


def embeddings(idx: int):
    print(senses[idx].stem)
    return np.load(senses[idx])


def conceptor(idx: int, aperture: float):
    return Conceptor.from_state_matrix(embeddings(idx), aperture)


def conceptor2d(idx: int, aperture=1):
    emb = embeddings(idx)
    emb2d = umap.UMAP().fit_transform(emb)
    return Conceptor.from_state_matrix(emb2d, aperture)


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


# %% Load data
print("Embeddings...")
emb_music = embeddings(9)
emb_ship = embeddings(10)
emb_arrow = embeddings(11)
emb_all = np.concatenate((emb_music, emb_ship, emb_arrow))


def emb_class(idx):
    if idx < len(emb_music):
        return "music"
    if idx < len(emb_music) + len(emb_ship):
        return "ship"
    return "arrow"


emb_classes = [emb_class(idx) for idx in range(len(emb_all))]

aperture = 0.07

print("Conceptors...")
co_music = Conceptor.from_state_matrix(emb_music, aperture)
co_ship = Conceptor.from_state_matrix(emb_ship, aperture)
co_arrow = Conceptor.from_state_matrix(emb_arrow, aperture)

co2_music = Conceptor.from_state_matrix(emb_music[:, :2], 0.07)
co2_ship = Conceptor.from_state_matrix(emb_ship[:, :2], 0.07)
co2_arrow = Conceptor.from_state_matrix(emb_arrow[:, :2], 0.07)
co3_music = Conceptor.from_state_matrix(emb_music[:, :3], 0.07)
co3_ship = Conceptor.from_state_matrix(emb_ship[:, :3], 0.07)
co3_arrow = Conceptor.from_state_matrix(emb_arrow[:, :3], 0.07)

# %% UMAP scatter
colors = {
    "music": "red",
    "ship": "blue",
    "arrow": "green",
}
plt.scatter(emb2d_all[:, 0], emb2d_all[:, 1], c=[colors[cl] for cl in emb_classes])

# %% Conceptors
plot_ellipses(co_music, co_ship, co_arrow)


# %%
def weighted_eigsum(diff: np.ndarray):
    eigs = np.linalg.eigvalsh(diff)
    return eigs.sum() / len(eigs)


def posneg_fraction(diff: Conceptor, tol=1e-3):
    eigs = np.sort(np.linalg.eigvals(diff))
    neg_amount = np.argwhere(eigs < -tol)[-1]
    pos_cutoff = np.argwhere(eigs > tol)[0]
    pos_amount = len(eigs) - pos_cutoff

    print(pos_amount, neg_amount, len(eigs) - (pos_amount + neg_amount))

    return pos_amount / (neg_amount + pos_amount)


# %%
eigvals = np.linalg.eigvals


def sigvals(x: np.ndarray):
    return np.linalg.svd(x)[1]


# %%
def plwf(*pts, **kwargs):
    projection = "2d" if len(pts) == 2 else "3d"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    ax.plot_wireframe(*pts, rstride=4, cstride=4, **kwargs)
    return ax


def wf(ax, col, pts):
    ax.plot_wireframe(*pts, rstride=4, cstride=4, color=col)


# %%
