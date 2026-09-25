"""
plotlib.py
Ve learning curve real-time bang Matplotlib, chay duoc trong plain script
(khong can Jupyter).

Fix quan trong: ban goc dung `from IPython import display` + `display.display(...)`
- day la co che chi danh cho Jupyter Notebook. Khi chay bang `python agent.py`
(script thuong), no khong ve gi huu ich (chi in ra repr "Figure(...)"), va viec
goi lien tuc `plt.pause()` cung luc voi Pygame dang bom Windows message loop
(`pygame.event.get()` trong play_step) co the gay xung dot GIL o tang native,
dan den crash "Fatal Python error: PyEval_RestoreThread" tren Windows.

Ban nay dung mot figure/axis co dinh, khong phu thuoc IPython, an toan hon
khi chay song song voi Pygame.
"""

import matplotlib.pyplot as plt

plt.ion()
_fig, _ax = plt.subplots()


def plot(scores, mean_scores):
    _ax.clear()
    _ax.set_title("Training")
    _ax.set_xlabel("Number of Games")
    _ax.set_ylabel("Score")
    _ax.plot(scores)
    _ax.plot(mean_scores)
    _ax.set_ylim(ymin=0)

    if scores:
        _ax.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        _ax.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)