from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from predictor import NUMERIC_COLS, STATIC_INPUT_COLS, TwoStagePulloutPredictor


APP_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False)) or os.environ.get("PULLOUT_PORTABLE") == "1"
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
BUNDLED_MODEL_PATH = RESOURCE_DIR / "artifacts" / "two_stage_pullout_model.pkl"
USER_MODEL_PATH = (
    Path(os.environ.get("LOCALAPPDATA", APP_DIR)) / "UHPCPulloutPredictor" / "two_stage_pullout_model.pkl"
    if IS_FROZEN
    else BUNDLED_MODEL_PATH
)
DEFAULT_DATASET = Path(
    os.environ.get("PULLOUT_DATASET", "__dataset_not_selected__.xlsx" if IS_FROZEN else r"E:\codezhangqiao\codex\数据库.xlsx")
)

FIELD_CONFIG = [
    ("Fiber shape", "Fiber shape", "choice", ["straight", "hooked end", "twisted", "corrugated"]),
    ("Diameter", "Diameter", "mm", "0.375"),
    ("Length", "Fiber length", "mm", "30"),
    ("Aspect ratio Lf/Df", "Aspect ratio L/D", "-", "80"),
    ("Embedded length", "Embedded length", "mm", "6.5"),
    ("Angle of inclination", "Inclination angle", "degree", "0"),
    ("Tensile strength", "Fiber tensile strength", "MPa", "2788"),
    ("Fiber pull-out Specimen shape", "Specimen shape", "choice", ["half dog-bone", "dog-bone", "prism", "cylinder"]),
    ("Loading rate", "Loading rate", "mm/s", "0.018"),
    ("W/B", "Water-to-binder ratio", "-", "0.18"),
    ("Compressive strength", "UHPC compressive strength", "MPa", "161"),
    ("Curing age", "Curing age", "day", "28"),
]


class PulloutPredictionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("UHPC Steel Fiber Pull-Out Response Predictor")
        self.geometry("1180x760")
        self.minsize(1050, 680)
        self.configure(bg="#eef2f7")
        self.model = None
        self.last_prediction = None
        self.dataset_path = DEFAULT_DATASET
        self.variables = {}
        self._configure_style()
        self._build_ui()
        self.after(150, self._initialize_model)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure("Title.TLabel", background="#0b2948", foreground="white", font=("Segoe UI", 19, "bold"))
        style.configure("Subtitle.TLabel", background="#0b2948", foreground="#c8d8e8", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="white", foreground="#0b2948", font=("Segoe UI", 11, "bold"))
        style.configure("Field.TLabel", background="white", foreground="#2d3c4d", font=("Segoe UI", 9))
        style.configure("Unit.TLabel", background="white", foreground="#69798a", font=("Segoe UI", 9))
        style.configure("ResultName.TLabel", background="#f3f7fb", foreground="#526477", font=("Segoe UI", 10))
        style.configure("ResultValue.TLabel", background="#f3f7fb", foreground="#087e8b", font=("Segoe UI", 20, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Card.TFrame")
        header.pack(fill="x")
        title_wrap = tk.Frame(header, bg="#0b2948", height=92)
        title_wrap.pack(fill="x")
        title_wrap.pack_propagate(False)
        ttk.Label(title_wrap, text="UHPC Steel Fiber Pull-Out Response Predictor", style="Title.TLabel").pack(
            anchor="w", padx=28, pady=(18, 2)
        )
        ttk.Label(
            title_wrap,
            text="Physics-informed two-stage prediction of peak response and the complete load–slip curve",
            style="Subtitle.TLabel",
        ).pack(anchor="w", padx=29)

        body = ttk.Frame(self, style="App.TFrame", padding=18)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=390)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        input_card = ttk.Frame(body, style="Card.TFrame", padding=18)
        input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(input_card, text="Input parameters", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        input_card.columnconfigure(1, weight=1)
        for row, (key, label, unit, default) in enumerate(FIELD_CONFIG, start=1):
            ttk.Label(input_card, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=default[0] if unit == "choice" else default)
            self.variables[key] = variable
            if unit == "choice":
                widget = ttk.Combobox(input_card, textvariable=variable, values=default, state="readonly", width=21)
            else:
                widget = ttk.Entry(input_card, textvariable=variable, width=23)
            widget.grid(row=row, column=1, sticky="ew", padx=(12, 8), pady=4)
            ttk.Label(input_card, text="" if unit == "choice" else unit, style="Unit.TLabel").grid(
                row=row, column=2, sticky="w", pady=4
            )

        ttk.Button(input_card, text="Calculate L/D", command=self._calculate_aspect).grid(
            row=len(FIELD_CONFIG) + 1, column=1, sticky="e", pady=(8, 2)
        )
        button_row = ttk.Frame(input_card, style="Card.TFrame")
        button_row.grid(row=len(FIELD_CONFIG) + 2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        self.predict_button = ttk.Button(
            button_row, text="Predict response", style="Primary.TButton", command=self._predict, state="disabled"
        )
        self.predict_button.pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Reset", command=self._reset).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Dataset…", command=self._choose_dataset).pack(side="left", padx=(8, 0))

        result_card = ttk.Frame(body, style="Card.TFrame", padding=18)
        result_card.grid(row=0, column=1, sticky="nsew")
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(3, weight=1)
        ttk.Label(result_card, text="Prediction results", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        metrics = tk.Frame(result_card, bg="#f3f7fb", padx=16, pady=12)
        metrics.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        metrics.columnconfigure((0, 1, 2), weight=1)
        ttk.Label(metrics, text="Peak load", style="ResultName.TLabel").grid(row=0, column=0)
        ttk.Label(metrics, text="Peak slip", style="ResultName.TLabel").grid(row=0, column=1)
        ttk.Label(metrics, text="Curve points", style="ResultName.TLabel").grid(row=0, column=2)
        self.pmax_label = ttk.Label(metrics, text="— N", style="ResultValue.TLabel")
        self.pmax_label.grid(row=1, column=0, pady=(4, 0))
        self.slip_label = ttk.Label(metrics, text="— mm", style="ResultValue.TLabel")
        self.slip_label.grid(row=1, column=1, pady=(4, 0))
        self.points_label = ttk.Label(metrics, text="—", style="ResultValue.TLabel")
        self.points_label.grid(row=1, column=2, pady=(4, 0))

        self.status_var = tk.StringVar(value="Preparing model…")
        self.status_label = ttk.Label(result_card, textvariable=self.status_var, foreground="#526477")
        self.status_label.grid(row=2, column=0, sticky="w", pady=(0, 6))

        plot_frame = ttk.Frame(result_card, style="Card.TFrame")
        plot_frame.grid(row=3, column=0, sticky="nsew")
        self.figure = Figure(figsize=(6.8, 4.4), dpi=100, facecolor="white")
        self.axes = self.figure.add_subplot(111)
        self._draw_empty_plot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

        action_row = ttk.Frame(result_card, style="Card.TFrame")
        action_row.grid(row=4, column=0, sticky="e", pady=(10, 0))
        self.export_button = ttk.Button(action_row, text="Export curve CSV", command=self._export, state="disabled")
        self.export_button.pack(side="right")
        ttk.Button(action_row, text="Save plot", command=self._save_plot).pack(side="right", padx=(0, 8))

    def _draw_empty_plot(self) -> None:
        self.axes.clear()
        self.axes.set_xlabel("Slip (mm)")
        self.axes.set_ylabel("Pull-out load (N)")
        self.axes.set_xlim(0, 5)
        self.axes.set_ylim(bottom=0)
        self.axes.grid(True, alpha=0.22, linewidth=0.8)
        self.axes.spines[["top", "right"]].set_visible(False)
        self.figure.tight_layout()

    def _initialize_model(self, force_retrain: bool = False) -> None:
        if IS_FROZEN and not force_retrain:
            model_path = USER_MODEL_PATH if USER_MODEL_PATH.exists() else BUNDLED_MODEL_PATH
            self.predict_button.configure(state="disabled")
            self.status_var.set("Loading the embedded prediction model…")

            def load_embedded() -> None:
                try:
                    model = TwoStagePulloutPredictor.load(model_path)
                    self.after(0, lambda: self._model_ready(model, True))
                except Exception as exc:
                    self.after(0, lambda: self._model_failed(exc))

            threading.Thread(target=load_embedded, daemon=True).start()
            return
        if not self.dataset_path.exists():
            self.status_var.set("Dataset not found. Select 数据库.xlsx to train the model.")
            return
        self.predict_button.configure(state="disabled")
        self.status_var.set("Training the two-stage model…" if force_retrain else "Loading the prediction model…")

        def work() -> None:
            try:
                use_cache = USER_MODEL_PATH.exists() and USER_MODEL_PATH.stat().st_mtime >= self.dataset_path.stat().st_mtime
                model = TwoStagePulloutPredictor.load(USER_MODEL_PATH) if use_cache and not force_retrain else None
                if model is None:
                    model = TwoStagePulloutPredictor().fit(self.dataset_path)
                    model.save(USER_MODEL_PATH)
                self.after(0, lambda: self._model_ready(model, use_cache and not force_retrain))
            except Exception as exc:
                self.after(0, lambda: self._model_failed(exc))

        threading.Thread(target=work, daemon=True).start()

    def _model_ready(self, model: TwoStagePulloutPredictor, from_cache: bool) -> None:
        self.model = model
        self.predict_button.configure(state="normal")
        source = "cached model" if from_cache else "newly trained model"
        self.status_var.set(f"Ready · {model.model_name} · {source}")

    def _model_failed(self, error: Exception) -> None:
        self.status_var.set("Model initialization failed.")
        messagebox.showerror("Model error", f"Could not initialize the prediction model:\n\n{error}")

    def _values(self) -> dict:
        values = {}
        for key in STATIC_INPUT_COLS:
            raw = self.variables[key].get().strip()
            if key in NUMERIC_COLS:
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{key} must be a number.") from exc
                if value <= 0 and key not in {"Angle of inclination"}:
                    raise ValueError(f"{key} must be greater than zero.")
                if key == "Angle of inclination" and not 0 <= value < 90:
                    raise ValueError("Angle of inclination must be in [0, 90) degrees.")
                values[key] = value
            else:
                values[key] = raw
        return values

    def _predict(self) -> None:
        if self.model is None:
            return
        try:
            values = self._values()
            ratio = values["Length"] / values["Diameter"]
            ratio_gap = abs(values["Aspect ratio Lf/Df"] - ratio) / max(ratio, 1e-8)
            prediction = self.model.predict(values)
            if ratio_gap > 0.05:
                prediction.warnings.insert(0, f"Aspect ratio differs from Length/Diameter ({ratio:.2f}) by more than 5%.")
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.last_prediction = prediction
        self.pmax_label.configure(text=f"{prediction.pmax:.2f} N")
        self.slip_label.configure(text=f"{prediction.slip:.3f} mm")
        self.points_label.configure(text=str(len(prediction.displacement)))
        self.axes.clear()
        self.axes.plot(prediction.displacement, prediction.load, color="#087e8b", linewidth=2.4, label="Predicted curve")
        self.axes.scatter([prediction.slip], [prediction.pmax], color="#e4572e", s=42, zorder=3, label="Predicted peak")
        self.axes.axvline(prediction.slip, color="#e4572e", linewidth=1, linestyle="--", alpha=0.5)
        self.axes.set_xlabel("Slip (mm)")
        self.axes.set_ylabel("Pull-out load (N)")
        self.axes.set_xlim(float(prediction.displacement.min()), float(prediction.displacement.max()))
        self.axes.set_ylim(bottom=0)
        self.axes.grid(True, alpha=0.22, linewidth=0.8)
        self.axes.spines[["top", "right"]].set_visible(False)
        self.axes.legend(frameon=False, loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.export_button.configure(state="normal")
        if prediction.warnings:
            self.status_var.set(f"Prediction complete · {len(prediction.warnings)} applicability warning(s)")
            messagebox.showwarning("Applicability domain", "\n".join(prediction.warnings))
        else:
            self.status_var.set("Prediction complete · all inputs are within the training-data ranges")

    def _calculate_aspect(self) -> None:
        try:
            length = float(self.variables["Length"].get())
            diameter = float(self.variables["Diameter"].get())
            self.variables["Aspect ratio Lf/Df"].set(f"{length / diameter:.3f}")
        except (ValueError, ZeroDivisionError):
            messagebox.showerror("Invalid input", "Enter valid non-zero length and diameter values first.")

    def _reset(self) -> None:
        for key, _, unit, default in FIELD_CONFIG:
            self.variables[key].set(default[0] if unit == "choice" else default)
        self.last_prediction = None
        self.pmax_label.configure(text="— N")
        self.slip_label.configure(text="— mm")
        self.points_label.configure(text="—")
        self.export_button.configure(state="disabled")
        self._draw_empty_plot()
        self.canvas.draw_idle()

    def _choose_dataset(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select the training database", filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")]
        )
        if selected:
            self.dataset_path = Path(selected)
            self._initialize_model(force_retrain=True)

    def _export(self) -> None:
        if self.last_prediction is None:
            return
        selected = filedialog.asksaveasfilename(
            title="Export predicted curve", defaultextension=".csv", filetypes=[("CSV file", "*.csv")]
        )
        if selected:
            frame = pd.DataFrame(
                {"Slip_mm": self.last_prediction.displacement, "Predicted_pullout_load_N": self.last_prediction.load}
            )
            frame.to_csv(selected, index=False, encoding="utf-8-sig")
            self.status_var.set(f"Curve exported to {selected}")

    def _save_plot(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save prediction plot", defaultextension=".png", filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")]
        )
        if selected:
            self.figure.savefig(selected, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    try:
        PulloutPredictionApp().mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
