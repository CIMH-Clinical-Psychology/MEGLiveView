#!/usr/bin/env python3
"""
Live MEG viewer for a (real or simulated) FieldTrip buffer
---------------------------------------------------------

* Streams continuously and **keeps all received samples** in memory (no ring
  overwrite).  Buffer is reset every time you press **Start**.
* Optional down‑sampling (default 100 Hz) plus selectable NumPy dtype
  (float64/32/16) minimise RAM consumption.  These two settings are only
  editable while the viewer is stopped.
* Current Python‑process memory footprint is displayed live in the status bar.
* Everything else (spectrogram, topography, on‑the‑fly parameter changes) works
  as before, now using the *post‑resample* sampling rate.

Dependencies
------------
`pip install numpy scipy mne matplotlib pyqt5 psutil`   # or pyside6 instead of pyqt5

Author:  Simon Kern <simon.kern@zi‑mannheim.de>
Date   : 2025‑05‑14
Licence: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import psutil
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Qt -------------------------------------------------------------
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
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
except ModuleNotFoundError:  # pragma: no cover
    from PySide6.QtCore import Qt, QTimer  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QApplication,
        QComboBox,
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
import mne  # noqa: E402
from scipy import interpolate  # noqa: E402

# External helper ----------------------------------------------------
from meg_utils.sigproc import resample  # noqa: E402 – provided by user

# --- FieldTrip buffer client ---------------------------------------
from externals.FieldTrip import FieldTripClient, FieldTripClientSimulator  # noqa: E402


# -------------------------------------------------------------------
# GUI main window
# -------------------------------------------------------------------
class LiveMEGViewer(QMainWindow):
    def __init__(self, client, sfreq_orig: float, ch_names: Sequence[str]):
        super().__init__()
        self.setWindowTitle("Live MEG viewer – FieldTrip buffer")
        self.client = client
        self.sfreq_orig = sfreq_orig  # Hz of incoming stream (≈1000)
        self.ch_names = list(ch_names)

        # ------------- mutable parameters --------------------------
        self.view_sec = -1          # unlimited history view
        self.win_sec = 2.0          # taper window length (s)
        self.fmin = 1.0
        self.fmax = 40.0
        self.update_freq = 1.0      # GUI refresh rate (s)
        self.target_sfreq = 100.0   # resample Hz – editable only stopped
        self.dtype_str = "float32"  # likewise

        # data buffer = list of 2‑D ndarrays (n_chan × n_samples)
        self._buf: List[np.ndarray] = []

        # timer is created in start_stream()
        self.timer: QTimer | None = None

        # ------------ UI layout -----------------------------------
        central = QWidget(); self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal); main_h.addWidget(splitter)

        # spectrogram pane (left) ----------------------------------
        self.fig_spec = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        splitter.addWidget(self.canvas_spec)

        # right side (topo + controls) -----------------------------
        right = QWidget(); right_v = QVBoxLayout(right); splitter.addWidget(right)
        self.fig_topo = Figure(figsize=(4, 3))
        self.canvas_topo = FigureCanvas(self.fig_topo); right_v.addWidget(self.canvas_topo, 3)
        self._build_controls(right_v)

        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2)

        # status bar with live memory usage ------------------------
        self.status = self.statusBar()
        self._update_memory_usage()

        # empty initial plots
        self._rebuild_spec_axes()
        self._clear_topo()

    # ------------------------------------------------------------------
    # Controls UI
    # ------------------------------------------------------------------
    def _build_controls(self, parent_layout):
        box = QGroupBox("Settings"); parent_layout.addWidget(box, 1)
        form = QFormLayout(box)

        # channel multi‑select
        self.list_channels = QListWidget(); self.list_channels.setSelectionMode(QListWidget.MultiSelection)
        for name in self.ch_names:
            QListWidgetItem(name, self.list_channels)
        self.list_channels.itemSelectionChanged.connect(self._on_params_cheap)
        form.addRow("Channel(s)", self.list_channels)

        # freq band
        self.spin_fmin = QDoubleSpinBox(minimum=0.0, maximum=500.0, singleStep=0.5, value=self.fmin)
        self.spin_fmax = QDoubleSpinBox(minimum=0.5, maximum=500.0, singleStep=0.5, value=self.fmax)
        self.spin_fmin.valueChanged.connect(self._on_params_cheap)
        self.spin_fmax.valueChanged.connect(self._on_params_cheap)
        form.addRow("fmin (Hz)", self.spin_fmin)
        form.addRow("fmax (Hz)", self.spin_fmax)

        # window / refresh / history seconds
        self.spin_win  = QDoubleSpinBox(minimum=0.1, maximum=10.0, singleStep=0.1, value=self.win_sec)
        self.spin_upd  = QDoubleSpinBox(minimum=0.1, maximum=5.0, singleStep=0.1, value=self.update_freq)
        self.spin_view = QSpinBox(minimum=-1, maximum=600, value=self.view_sec)
        self.spin_win.valueChanged.connect(self._on_params_cheap)
        self.spin_upd.valueChanged.connect(self._on_update_freq_changed)
        self.spin_view.valueChanged.connect(self._on_params_cheap)
        form.addRow("window_s",   self.spin_win)
        form.addRow("update_s",   self.spin_upd)
        form.addRow("view_s (‑1 = all)", self.spin_view)

        # resample & dtype (disabled while running)
        self.spin_rsamp = QDoubleSpinBox(minimum=10.0, maximum=self.sfreq_orig, singleStep=10.0, value=self.target_sfreq)
        self.combo_dtype = QComboBox(); self.combo_dtype.addItems(["float64", "float32", "float16"])
        self.combo_dtype.setCurrentText(self.dtype_str)
        form.addRow("resample Hz", self.spin_rsamp)
        form.addRow("dtype",       self.combo_dtype)

        # start / stop buttons
        btns = QHBoxLayout(); form.addRow(btns)
        self.btn_start = QPushButton("Start"); self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_stream); self.btn_stop.clicked.connect(self.stop_stream)
        btns.addWidget(self.btn_start); btns.addWidget(self.btn_stop)

    # ------------------------------------------------------------------
    # Start / Stop streaming
    # ------------------------------------------------------------------
    def start_stream(self):
        if self.timer and self.timer.isActive():
            return
        # grab resample/dtype settings and lock widgets
        self.target_sfreq = self.spin_rsamp.value()
        self.dtype_str = self.combo_dtype.currentText()
        self.spin_rsamp.setEnabled(False)
        self.combo_dtype.setEnabled(False)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        # reset buffer
        self._buf.clear(); self._rebuild_spec_axes(); self._clear_topo()
        # timer
        self.timer = QTimer(self); self.timer.timeout.connect(self._update_cycle)
        self.timer.start(int(self.update_freq * 1000))

    def stop_stream(self):
        if self.timer: self.timer.stop()
        self.spin_rsamp.setEnabled(True); self.combo_dtype.setEnabled(True)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Param changes (cheap: no redraw needed)
    # ------------------------------------------------------------------
    def _on_params_cheap(self):
        self.fmin  = self.spin_fmin.value(); self.fmax = self.spin_fmax.value()
        self.win_sec = self.spin_win.value(); self.view_sec = self.spin_view.value()
        self._rebuild_spec_axes()

    def _on_update_freq_changed(self):
        self.update_freq = self.spin_upd.value()
        if self.timer and self.timer.isActive():
            self.timer.stop(); self.timer.start(int(self.update_freq * 1000))
        self._on_params_cheap()

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------
    def _update_cycle(self):
        try:
            new_raw, _ = self.client.get_data(chunksize=1000)  # n_chan × 1000 @ sfreq_orig
        except Exception as exc:
            print("Data fetch error:", exc); self.stop_stream(); return

        # --- resample & dtype cast --------------------------------
        new_ds = resample(new_raw, self.sfreq_orig, self.target_sfreq, n_jobs=1, verbose=False)
        new_ds = new_ds.astype(self.dtype_str, copy=False)
        self._buf.append(new_ds)

        # optional view limit (‑1 = unlimited)
        if self.view_sec > 0:
            max_samples = int(self.view_sec * self.target_sfreq)
            # trim from head until under limit
            cur = sum(b.shape[1] for b in self._buf)
            while cur > max_samples and self._buf:
                cur -= self._buf[0].shape[1]; self._buf.pop(0)

        # build one big 2‑D array for plotting
        data_buf = np.concatenate(self._buf, axis=1) if self._buf else None
        if data_buf is None or data_buf.shape[1] < 2:
            return

        self._plot_spectrogram(data_buf)
        self._plot_topography(data_buf)
        self._update_memory_usage()

    # ------------------------------------------------------------------
    # Spectrogram (multitaper) -----------------------------------------
    # ------------------------------------------------------------------
    def _rebuild_spec_axes(self):
        sel = self._selected_ch_indices() or [0]
        n_rows = len(sel)
        self.fig_spec.clf(); self.axes_spec = []; self.spec_imgs = []
        for i in range(n_rows):
            ax = self.fig_spec.add_subplot(n_rows, 1, i + 1)
            if i != n_rows - 1: ax.set_xticklabels([])
            self.axes_spec.append(ax); self.spec_imgs.append(None)
        self.fig_spec.suptitle("Multitaper spectrogram")
        self.canvas_spec.draw_idle()

    def _plot_spectrogram(self, data_buf: np.ndarray):
        sel = self._selected_ch_indices() or [0]
        if len(sel) != len(self.axes_spec):
            self._rebuild_spec_axes()
        nperseg = int(max(2, self.win_sec * self.target_sfreq))
        step = max(1, int(self.target_sfreq // (1 / self.win_sec)))
        time_pts = np.arange(data_buf.shape[1]) / self.target_sfreq
        freqs = np.arange(self.fmin, self.fmax + 1, 1)
        n_cycles = np.ones_like(freqs) * 2.0
        for row, ch_idx in enumerate(sel):
            sig = data_buf[ch_idx][np.newaxis, np.newaxis, :]
            power = mne.time_frequency.tfr_array_multitaper(
                sig, sfreq=self.target_sfreq, freqs=freqs,
                n_cycles=n_cycles, time_bandwidth=4.0, output="power",
                decim=step, verbose=False,
            )[0, 0]
            ax = self.axes_spec[row]
            if self.spec_imgs[row] is None:
                im = ax.imshow(power, aspect="auto", origin="lower",
                               extent=[time_pts[0], time_pts[-1], freqs[0], freqs[-1]],
                               cmap="viridis")
                self.spec_imgs[row] = im; ax.set_ylabel(self.ch_names[ch_idx])
                if row == len(sel) - 1: ax.set_xlabel("time (s)")
            else:
                self.spec_imgs[row].set_data(power)
                self.spec_imgs[row].set_extent([time_pts[0], time_pts[-1], freqs[0], freqs[-1]])
            ax.set_ylim(self.fmin, self.fmax)
        self.canvas_spec.draw_idle()

    # ------------------------------------------------------------------
    # Topography -------------------------------------------------------
    # ------------------------------------------------------------------
    def _clear_topo(self):
        self.fig_topo.clf(); ax = self.fig_topo.add_subplot(111)
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.axis("off")
        self.canvas_topo.draw_idle()

    def _plot_topography(self, data_buf: np.ndarray):
        psd = np.mean(data_buf ** 2, axis=1)
        self.fig_topo.clf(); ax = self.fig_topo.add_subplot(111)
        # simple square grid fallback
        n_ch = len(psd); n_rows = int(np.ceil(np.sqrt(n_ch))); n_cols = int(np.ceil(n_ch / n_rows))
        grid = np.full((n_rows, n_cols), np.nan); idx = np.unravel_index(np.arange(n_ch), (n_rows, n_cols))
        grid[idx] = psd
        im = ax.imshow(grid, origin="lower", aspect="equal", cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        self.fig_topo.colorbar(im, ax=ax, shrink=0.7, pad=0.05, label="⟨x²⟩")
        ax.set_title("Topography (mean power)")
        self.canvas_topo.draw_idle()

    # ------------------------------------------------------------------
    # Misc helpers -----------------------------------------------------
    # ------------------------------------------------------------------
    def _selected_ch_indices(self) -> List[int]:
        return [self.list_channels.row(it) for it in self.list_channels.selectedItems()]

    def _update_memory_usage(self):
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self.status.showMessage(f"Memory: {mem_mb: .1f} MB | dtype {self.dtype_str} @ {self.target_sfreq:g} Hz")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Live MEG viewer for FieldTrip buffer")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1972)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--sfreq", type=float, default=1000.0, help="simulator Hz")
    args = parser.parse_args(argv)

    if args.sim:
        client = FieldTripClientSimulator(sfreq=args.sfreq, nchan=306); info = client.get_measurement_info()
    else:
        try:
            client = FieldTripClientSimulator(hostname=args.host, port=args.port); info = client.get_measurement_info()
        except Exception as exc:
            print("Cannot connect:", exc, file=sys.stderr); sys.exit(1)

    sfreq_orig = info.get("sfreq", args.sfreq)
    ch_names = info.get("ch_names", [f"CH{i:03d}" for i in range(info["nchan"])])

    app = QApplication(sys.argv)
    viewer = LiveMEGViewer(client, sfreq_orig=sfreq_orig, ch_names=ch_names)
    viewer.resize(1400, 800); viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
