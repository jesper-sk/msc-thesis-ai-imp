# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.patches import Ellipse as MEllipse

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
        ellipse = item if isinstance(item, Ellipse) else item.to_ellipse()

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

print("Conceptors...")
co_music = Conceptor.from_state_matrix(emb_music)
co_ship = Conceptor.from_state_matrix(emb_ship)
co_arrow = Conceptor.from_state_matrix(emb_arrow)

print("UMAP dimensionality reduction...")
fitter = umap.UMAP()
emb2d_all = fitter.fit_transform(emb_all)
emb2d_music = fitter.transform(emb_music)
emb2d_ship = fitter.transform(emb_ship)
emb2d_arrow = fitter.transform(emb_arrow)

print("2D conceptors from UMAP...")
co2d_music = Conceptor.from_state_matrix(emb2d_music)
co2d_ship = Conceptor.from_state_matrix(emb2d_ship)
co2d_arrow = Conceptor.from_state_matrix(emb2d_arrow)

# %% UMAP scatter
colors = {
    "music": "red",
    "ship": "blue",
    "arrow": "green",
}
plt.scatter(emb2d_all[:, 0], emb2d_all[:, 1], c=[colors[cl] for cl in emb_classes])

# %% Conceptors
plot_ellipses(co2d_music, co2d_ship, co2d_arrow)


# %%
def bricman(x: Conceptor, y: Conceptor):
    diff = x - y
    eigs = np.linalg.eigvalsh(diff)
    return eigs.sum() / len(eigs)


def test(diff: Conceptor, tol=1e-3):
    eigs = np.sort(np.linalg.eigvals(diff))
    neg_amount = np.argwhere(eigs < -tol)[-1]
    pos_amount = len(eigs) - np.argwhere(eigs > tol)[0]
    return pos_amount / (neg_amount + pos_amount)


# %%
eigvals = np.linalg.eigvals
# %%
