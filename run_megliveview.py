#!/usr/bin/env python3
"""
Live MEG viewer for a (real or simulated) FieldTrip buffer
---------------------------------------------------------

GUI layout (see sketch):
  ┌──────────────────────────────────────────────────────┐
  │ left:  Spectrogram of user‑selected channel(s)       │
  │       (one below the other if >1)                    │
  ├────────────────────────────────┬─────────────────────┤
  │ top‑right: topography (all ch) │ bottom‑right: ctrl  │
  │                                │  widgets / params   │
  └────────────────────────────────┴─────────────────────┘

Main features
-------------
* Pulls new data every *update_freq* seconds via ``FieldTripClient.get_data``
  (or ``FieldTripClientSimulator`` when ``--sim`` is passed).
* Ring‑buffer keeps either *view_sec* seconds or *all* streamed samples.
* Per‑channel multitaper spectrogram using ``mne.time_frequency.tfr_array_multitaper``.
* "Topography" = simple 2‑D interpolation of the same power values; it will
  fall back to a square grid when true sensor positions are unavailable.
* Widgets let the user change frequency band, taper window length, update
  rate, amount of data to keep, and the channels to display. Any change is
  applied immediately (timer restarts, figures clear, buffer re‑sized …).
* Designed for >10‑minute runs at 1 kHz × 306 ch without leaking memory—
  figures are updated in‑place and NumPy ring buffers reuse existing memory.

Dependencies
------------
``pip install numpy scipy mne matplotlib pyqt5``  # or pyside6 instead of pyqt5

Author:  Simon Kern <simon.kern@zi-mannheim.de>
Date   :  2025‑05‑14
Licence:  MIT
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import List, Sequence

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

# --- Qt -------------------------------------------------------------
try:  # prefer PyQt5, fall back to PySide6
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError:  # pragma: no cover – only reached when PyQt5 absent
    from PySide6.QtCore import Qt, QTimer  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

# --- Signal processing ---------------------------------------------
import mne  # noqa: E402  (heavy import – keep after Qt for quicker GUI launch)
from scipy import interpolate  # noqa: E402

# --- FieldTrip buffer client ---------------------------------------
from externals.FieldTrip import FieldTripClient, FieldTripClientSimulator  # noqa: E402


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def best_grid(n: int) -> tuple[int, int]:
    """Return (n_rows, n_cols) forming the *smallest* square/rectangular grid ≥ *n*."""
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def ring_append(ring: deque, arr: np.ndarray, maxlen: int | None):
    """Append *arr* along *axis=1* to a ring‑buffer *deque* while respecting *maxlen* samples."""
    ring.extend(np.array_split(arr, arr.shape[1], axis=1))  # 1‑sample chunks
    if maxlen is not None:
        while len(ring) > maxlen:
            ring.popleft()


# -------------------------------------------------------------------
# GUI main window
# -------------------------------------------------------------------
class LiveMEGViewer(QMainWindow):
    def __init__(self, client, sfreq: float, ch_names: Sequence[str]):
        super().__init__()
        self.setWindowTitle("Live MEG viewer – FieldTrip buffer")
        self.client = client
        self.sfreq = sfreq
        self.ch_names = list(ch_names)

        # ------------ state -------------
        self.view_sec = -1
        self.update_freq = 1.0  # s
        self.win_sec = 2.0
        self.fmin = 1.0
        self.fmax = 40.0
        self.resample = 0
        self.timer: QTimer | None = None

        # ring buffer: deque of (n_chan × 1) samples, each entry shape (n_chan, 1)
        self._ring: deque[np.ndarray] = deque()

        # ------------ UI -------------
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        h_layout.addWidget(splitter)

        # left plot – spectrogram ------------------------------------------------
        self.fig_left = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas_left = FigureCanvas(self.fig_left)
        splitter.addWidget(self.canvas_left)

        # right side = two plots + controls -------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        splitter.addWidget(right_container)

        # topography figure ------------------------------------------------------
        self.fig_topo = Figure(figsize=(4, 3))
        self.canvas_topo = FigureCanvas(self.fig_topo)
        right_layout.addWidget(self.canvas_topo, stretch=3)

        # controls ---------------------------------------------------------------
        self._build_controls(right_layout)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # initial plot placeholders
        self._init_plots()

    # ---------------------------------------------------------------------
    # Controls UI
    # ---------------------------------------------------------------------
    def _build_controls(self, parent_layout):
        box = QGroupBox("Settings")
        parent_layout.addWidget(box, stretch=1)
        form = QFormLayout(box)

        # channel list
        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QListWidget.MultiSelection)
        for name in self.ch_names:
            QListWidgetItem(name, self.list_channels)
        self.list_channels.itemSelectionChanged.connect(self._on_params_changed)
        form.addRow("Channel(s)", self.list_channels)

        # frequency band
        self.spin_fmin = QDoubleSpinBox(minimum=0.0, maximum=500.0, singleStep=0.5, value=self.fmin)
        self.spin_fmax = QDoubleSpinBox(minimum=0.5, maximum=500.0, singleStep=0.5, value=self.fmax)
        self.spin_fmin.valueChanged.connect(self._on_params_changed)
        self.spin_fmax.valueChanged.connect(self._on_params_changed)
        form.addRow("fmin (Hz)", self.spin_fmin)
        form.addRow("fmax (Hz)", self.spin_fmax)

        # window / update / view seconds
        self.spin_win = QDoubleSpinBox(minimum=0.1, maximum=10.0, singleStep=0.1, value=self.win_sec)
        self.spin_upd = QDoubleSpinBox(minimum=0.1, maximum=5.0, singleStep=0.1, value=self.update_freq)
        self.spin_view = QSpinBox(minimum=-1, maximum=600, value=self.view_sec)
        self.spin_win.valueChanged.connect(self._on_params_changed)
        self.spin_upd.valueChanged.connect(self._on_update_freq_changed)
        self.spin_view.valueChanged.connect(self._on_params_changed)
        form.addRow("window_s", self.spin_win)
        form.addRow("update_s", self.spin_upd)
        form.addRow("view_s (-1=all)", self.spin_view)

        # start/stop buttons
        btn_box = QHBoxLayout()
        form.addRow(btn_box)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_stop)

    # ------------------------------------------------------------------
    # Plot initialisation
    # ------------------------------------------------------------------
    def _init_plots(self):
        # left spec figure – keep empty axes list for dynamic channel rows
        self.axes_spec: list = []
        self.spec_imgs: list = []
        self.fig_left.clf()
        self.canvas_left.draw_idle()

        # topography placeholder
        self.fig_topo.clf()
        ax = self.fig_topo.add_subplot(111)
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax.axis("off")
        self.canvas_topo.draw_idle()

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _on_params_changed(self):
        self.fmin = self.spin_fmin.value()
        self.fmax = self.spin_fmax.value()
        self.win_sec = self.spin_win.value()
        self.view_sec = self.spin_view.value()
        # spectrogram axes need to be recreated if channel list changed
        self._rebuild_spec_axes()

    def _on_update_freq_changed(self):
        self.update_freq = self.spin_upd.value()
        if self.timer and self.timer.isActive():
            self.timer.stop()
            self.timer.start(int(self.update_freq * 1000))
        self._on_params_changed()

    # ------------------------------------------------------------------
    # Streaming control
    # ------------------------------------------------------------------
    def start_stream(self):
        if self.timer and self.timer.isActive():
            return  # already running
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_cycle)
        self.timer.start(int(self.update_freq * 1000))

    def stop_stream(self):
        if self.timer:
            self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------
    def _update_cycle(self):
        try:
            new_data, _ = self.client.get_data(chunksize=1000)  # (n_chan, n_samp)
        except Exception as exc:  # pragma: no cover – network errors etc.
            print("Error fetching data:", exc)
            self.stop_stream()
            return

        # ---- update ring buffer -----------------------------------
        ring_append(self._ring, new_data, None if self.view_sec < 0 else int(self.view_sec * self.sfreq))

        # build ndarray of current buffer (n_chan, n_samples)
        if not self._ring:
            return
        data_buf = np.concatenate(list(self._ring), axis=1)

        # ---- update plots -----------------------------------------
        self._plot_spectrogram(data_buf)
        self._plot_topography(data_buf)

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------
    def _rebuild_spec_axes(self):
        # clear & rebuild spec axes when channels or params change
        self.fig_left.clf()
        self.axes_spec.clear()
        self.spec_imgs.clear()

        sel_rows = self._selected_ch_indices()
        n_rows = max(1, len(sel_rows))
        for i in range(n_rows):
            ax = self.fig_left.add_subplot(n_rows, 1, i + 1)
            ax.set_ylabel(self.ch_names[sel_rows[i]] if sel_rows else "-")
            if i != n_rows - 1:
                ax.set_xticklabels([])
            self.axes_spec.append(ax)
            self.spec_imgs.append(None)
        self.fig_left.suptitle("Multitaper spectrogram")
        self.canvas_left.draw_idle()

    def _plot_spectrogram(self, data_buf: np.ndarray):
        sel = self._selected_ch_indices()
        if not sel:
            sel = [0]  # default first channel
        n_rows = len(sel)

        if len(self.axes_spec) != n_rows:
            self._rebuild_spec_axes()

        nperseg = int(self.win_sec * self.sfreq)
        if nperseg < 2:
            return
        step = max(1, int(self.sfreq // (1 / self.win_sec)))
        time_pts = np.arange(0, data_buf.shape[1]) / self.sfreq
        freqs = np.arange(self.fmin, self.fmax + 1, 1)
        n_cycles = np.ones_like(freqs) * 2.0  # keeps time‑freq trade‑off simple

        for row, ch_idx in enumerate(sel):
            sig = data_buf[ch_idx][np.newaxis, np.newaxis, :]
            try:
                power = mne.time_frequency.tfr_array_multitaper(
                    sig,
                    sfreq=self.sfreq,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    time_bandwidth=4.0,
                    output="power",
                    decim=step,
                    verbose=False,
                )[0, 0]
            except Exception as exc:
                print("tfr failed:", exc)
                continue
            ax = self.axes_spec[row]
            if self.spec_imgs[row] is None:
                extent = [time_pts[0], time_pts[-1], freqs[0], freqs[-1]]
                im = ax.imshow(
                    power,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                )
                ax.set_ylabel(self.ch_names[ch_idx])
                if row == n_rows - 1:
                    ax.set_xlabel("time (s)")
                self.spec_imgs[row] = im
            else:
                self.spec_imgs[row].set_data(power)
                self.spec_imgs[row].set_extent([time_pts[0], time_pts[-1], freqs[0], freqs[-1]])
            ax.set_ylim(self.fmin, self.fmax)

        self.canvas_left.draw_idle()

    # ------------------------------------------------------------------
    # Topography (very simple grid when proper positions unknown)
    # ------------------------------------------------------------------
    def _plot_topography(self, data_buf: np.ndarray):
        # average power in current band per channel
        psd = np.mean(data_buf ** 2, axis=1)
        n_ch = len(psd)
        self.fig_topo.clf()
        ax = self.fig_topo.add_subplot(111)

        # try to plot with real positions if available
        pos = self._sensor_positions_mne()
        if pos is not None and len(pos) == n_ch:
            im, cn = mne.viz.plot_topomap(
                psd,
                pos,
                axes=ax,
                show=False,
                cmap="viridis",
                contours=0,
            )
        else:
            # fallback: square grid
            n_rows, n_cols = best_grid(n_ch)
            grid = np.full((n_rows, n_cols), np.nan)
            idx = np.unravel_index(np.arange(n_ch), (n_rows, n_cols))
            grid[idx] = psd
            im = ax.imshow(grid, origin="lower", aspect="equal", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig_topo.colorbar(im, ax=ax, shrink=0.7, pad=0.05, label="⟨x²⟩")
        ax.set_title("Topography (mean power)")
        self.canvas_topo.draw_idle()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _selected_ch_indices(self) -> List[int]:
        sel_items = self.list_channels.selectedItems()
        return [self.list_channels.row(it) for it in sel_items]

    def _sensor_positions_mne(self):
        """Return 2‑D sensor positions if MNE info has them, else *None*."""
        try:
            info = self.client.get_measurement_info()
            if "chs" in info:  # real FieldTrip client provides MNE‑style info
                locs = [c["loc"][:3] for c in info["chs"]]  # type: ignore
                locs = np.array(locs)
                if np.all(np.isfinite(locs)):
                    # project 3‑D → 2‑D with azimuthal projection
                    return mne.channels.layout._topomap._make_equidistant_layout(locs)[0]
        except Exception:  # pragma: no cover
            pass
        return None


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Live MEG viewer for FieldTrip buffer")
    parser.add_argument("--host", default="localhost", help="FieldTrip buffer host (default: localhost)")
    parser.add_argument("--port", type=int, default=1972, help="port (default: 1972)")
    parser.add_argument("--sim", action="store_true", help="use FieldTripClientSimulator instead of real buffer")
    parser.add_argument("--sfreq", type=float, default=1000.0, help="simulator sampling rate (Hz)")
    args = parser.parse_args(argv)

    # --- connect --------------------------------------------------
    if args.sim:
        client = FieldTripClientSimulator(sfreq=args.sfreq, nchan=306)
        info = client.get_measurement_info()
    else:
        try:
            client = FieldTripClientSimulator(hostname=args.host, port=args.port)
            info = client.get_measurement_info()
        except Exception as exc:
            print("Cannot connect to FieldTrip buffer:", exc, file=sys.stderr)
            sys.exit(1)

    sfreq = info.get("sfreq", args.sfreq)
    ch_names = info.get("ch_names", [f"CH{i:03d}" for i in range(info["nchan"])])

    # --- Qt application -------------------------------------------
    app = QApplication(sys.argv)
    viewer = LiveMEGViewer(client, sfreq=sfreq, ch_names=ch_names)
    viewer.resize(1400, 800)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
