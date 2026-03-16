import io
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gamma, linregress, norm


st.set_page_config(page_title="SAXS WLC Explorer", layout="wide")
st.title("SAXS Worm-like Chain Explorer")
st.caption("Manual 2D polydispersity integration over contour length (L) and Kuhn length (b)")


try:
    from sasmodels.core import load_model
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel

    try:
        from sasmodels.direct_model import call_kernel
    except Exception:
        call_kernel = None

    SASMODELS_AVAILABLE = True
except Exception:
    SASMODELS_AVAILABLE = False


if not SASMODELS_AVAILABLE:
    st.error(
        "sasmodels is not installed or could not be imported. "
        "Install with: pip install sasmodels"
    )
    st.stop()


@dataclass
class DistSpec:
    name: str
    mean: float
    pd_index: float
    n_bins: int


@st.cache_resource(show_spinner=False)
def get_flexible_cylinder_model():
    return load_model("flexible_cylinder")


class SasmodelsEvaluator:
    def __init__(self, q: np.ndarray):
        self.q = np.asarray(q, dtype=float)
        self.model = get_flexible_cylinder_model()
        self.direct: Optional[DirectModel] = None
        self.engine_label = "unknown"
        self._init_direct_model()

    def _init_direct_model(self) -> None:
        try:
            data = empty_data1D(self.q)
            self.direct = DirectModel(data, self.model)
            self.engine_label = "DirectModel(data, loaded model)"
            return
        except Exception:
            self.direct = None

        # Keep compatibility across sasmodels versions that require model_info.
        try:
            from sasmodels.core import load_model_info

            data = empty_data1D(self.q)
            model_info = load_model_info("flexible_cylinder")
            self.direct = DirectModel(data, model_info)
            self.engine_label = "DirectModel(data, model_info)"
            return
        except Exception:
            self.direct = None

    def iq(self, params: Dict[str, float]) -> np.ndarray:
        direct_error = None
        if self.direct is not None:
            try:
                return np.asarray(self.direct(**params), dtype=float)
            except Exception as exc:
                direct_error = exc

        if call_kernel is not None:
            try:
                return np.asarray(call_kernel(self.model, params, self.q), dtype=float)
            except Exception as kernel_exc:
                raise RuntimeError(
                    f"DirectModel failed: {direct_error}; call_kernel failed: {kernel_exc}"
                ) from kernel_exc

        if direct_error is not None:
            raise RuntimeError(
                f"DirectModel failed and call_kernel unavailable in this sasmodels build: {direct_error}"
            ) from direct_error
        raise RuntimeError("No compatible sasmodels evaluation path found.")


def benoit_doty_rg(length: float, kuhn_length: float) -> float:
    """
    Benoit-Doty form for worm-like chain radius of gyration.
    b is Kuhn length (2*persistence length).
    """
    l = max(length, 1e-12)
    b = max(kuhn_length, 1e-12)
    x = l / b
    rg2 = (b * l / 6.0) - (b * b / 4.0) + (b * b / (4.0 * x)) - (b * b / (8.0 * x * x)) * (1.0 - np.exp(-2.0 * x))
    return float(np.sqrt(max(rg2, 0.0)))


def _quadrature_weights_from_pdf(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    pdf = np.clip(pdf, 0.0, None)
    if x.size == 1:
        return np.array([1.0], dtype=float)

    # Use local bin widths so non-uniform grids (e.g. log spacing) integrate correctly.
    dx = np.empty_like(x, dtype=float)
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    dx[0] = 0.5 * (x[1] - x[0])
    dx[-1] = 0.5 * (x[-1] - x[-2])
    dx = np.clip(dx, 1e-300, None)

    mass = pdf * dx
    s = np.sum(mass)
    if s <= 0:
        w = np.zeros_like(x, dtype=float)
        w[int(np.argmin(np.abs(x - np.mean(x))))] = 1.0
        return w
    return mass / s


def build_distribution(spec: DistSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    mean = max(float(spec.mean), eps)
    pd_idx = max(float(spec.pd_index), 0.0)
    n_bins = max(int(spec.n_bins), 1)

    if pd_idx == 0.0 or n_bins == 1:
        return (
            np.array([mean], dtype=float),
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
        )

    sigma = pd_idx * mean

    if spec.name == "Gaussian":
        low = max(eps, mean - 6.0 * sigma)
        high = mean + 6.0 * sigma
        if high <= low:
            high = low * 1.01
        x = np.linspace(low, high, n_bins)
        pdf = norm.pdf(x, loc=mean, scale=max(sigma, eps))

    elif spec.name == "Lognormal":
        cv = max(pd_idx, 1e-6)
        sigma_ln = np.sqrt(np.log(1.0 + cv * cv))
        mu_ln = np.log(mean) - 0.5 * sigma_ln * sigma_ln
        low = np.exp(mu_ln - 4.5 * sigma_ln)
        high = np.exp(mu_ln + 4.5 * sigma_ln)
        high = min(high, mean * 1e6)
        x = np.geomspace(max(low, eps), max(high, low * 1.01), n_bins)
        pdf = (1.0 / (x * sigma_ln * np.sqrt(2.0 * np.pi))) * np.exp(-((np.log(x) - mu_ln) ** 2) / (2.0 * sigma_ln**2))

    elif spec.name == "Schulz (Gamma)":
        k = max(1.0 / (pd_idx * pd_idx), 1e-3)
        theta = mean / k
        low = max(eps, gamma.ppf(1e-4, a=k, scale=theta))
        high = gamma.ppf(1.0 - 1e-4, a=k, scale=theta)
        high = max(high, low * 1.01)
        x = np.linspace(low, high, n_bins)
        pdf = gamma.pdf(x, a=k, scale=theta)

    elif spec.name == "Triangular":
        half_width = sigma * np.sqrt(6.0)
        low = max(eps, mean - half_width)
        high = mean + half_width
        if high <= low:
            high = low * 1.01
        mode = np.clip(mean, low, high)
        x = np.linspace(low, high, n_bins)
        left = np.where(mode > low, (x - low) / (mode - low + eps), 0.0)
        right = np.where(high > mode, (high - x) / (high - mode + eps), 0.0)
        pdf = np.where(x <= mode, left, right)

    elif spec.name == "Uniform":
        half_width = sigma * np.sqrt(3.0)
        low = max(eps, mean - half_width)
        high = mean + half_width
        if high <= low:
            high = low * 1.01
        x = np.linspace(low, high, n_bins)
        pdf = np.ones_like(x)

    elif spec.name == "Boltzmann":
        # Symmetric Boltzmann-like energy weighting around mean.
        kbt = max(sigma / np.sqrt(2.0), mean * 1e-6)
        low = max(eps, mean - 10.0 * kbt)
        high = mean + 10.0 * kbt
        if high <= low:
            high = low * 1.01
        x = np.linspace(low, high, n_bins)
        pdf = np.exp(-np.abs(x - mean) / kbt)

    else:
        raise ValueError(f"Unsupported distribution: {spec.name}")

    # Probability density for plotting.
    area = float(np.trapezoid(pdf, x)) if x.size > 1 else float(pdf[0])
    if area > 0:
        pdf = pdf / area
    else:
        pdf = np.zeros_like(x, dtype=float)
        pdf[int(np.argmin(np.abs(x - mean)))] = 1.0

    # Quadrature weights for discrete integration.
    w = _quadrature_weights_from_pdf(x, pdf)
    return x, pdf, w


def compute_intensity(evaluator: SasmodelsEvaluator, params: Dict[str, float]) -> np.ndarray:
    return evaluator.iq(params)


def weighted_2d_polydisperse_iq(
    evaluator: SasmodelsEvaluator,
    q: np.ndarray,
    base_params: Dict[str, float],
    l_grid: np.ndarray,
    l_w: np.ndarray,
    b_grid: np.ndarray,
    b_w: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(q, dtype=float)
    radius = float(base_params["radius"])
    scale_total = float(base_params.get("scale", 1.0))
    background = float(base_params.get("background", 0.0))

    # Evaluate per-component intensity with unit scale and zero background,
    # then compose with volume-fraction weights. This preserves low-q scaling
    # for polydisperse form factors when "scale" is total volume fraction.
    component_base = dict(base_params)
    component_base["scale"] = 1.0
    component_base["background"] = 0.0

    l_volumes = np.pi * (radius**2) * l_grid
    joint_nw = np.outer(l_w, b_w)
    joint_vw = joint_nw * l_volumes[:, None]
    vw_sum = float(np.sum(joint_vw))
    if vw_sum <= 0:
        joint_vw = joint_nw
        vw_sum = float(np.sum(joint_vw))
    if vw_sum > 0:
        joint_vw = joint_vw / vw_sum

    for i, (l_val, wl) in enumerate(zip(l_grid, l_w)):
        if wl <= 0:
            continue
        for j, b_val in enumerate(b_grid):
            w = float(joint_vw[i, j])
            if w <= 0:
                continue
            pars = dict(component_base)
            pars["length"] = float(l_val)
            pars["kuhn_length"] = float(b_val)
            out += w * compute_intensity(evaluator, pars)
    return scale_total * out + background


def format_metadata(metadata: Dict[str, float]) -> str:
    lines = [f"# {k}: {v}" for k, v in metadata.items()]
    return "\n".join(lines) + "\n"


def format_parameter_dump(metadata: Dict[str, float]) -> str:
    """
    Deterministic plain-text dump for testing/replay.
    Keys are sorted for stable file diffs.
    """
    lines = ["[saxs_wlc_parameters]"]
    for key in sorted(metadata.keys()):
        val = metadata[key]
        if isinstance(val, float):
            lines.append(f"{key}={val:.15g}")
        else:
            lines.append(f"{key}={val}")
    return "\n".join(lines) + "\n"


def dump_parameters_txt(metadata: Dict[str, float], path: str) -> None:
    text = format_parameter_dump(metadata)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


with st.sidebar:
    st.header("Q-Range")
    q_min = st.number_input("q_min", min_value=1e-6, value=1e-3, format="%.6g")
    q_max = st.number_input("q_max", min_value=1e-6, value=5e-1, format="%.6g")
    q_points = st.number_input("N_q", min_value=10, max_value=10000, value=300, step=10)
    q_log = st.toggle("Log spacing", value=True)

    st.header("Physical Properties")
    radius = st.number_input("Radius R", min_value=1e-6, value=20.0, format="%.6g")
    sld = st.number_input("SLD Polymer", value=1.0, format="%.6g")
    sld_solvent = st.number_input("SLD Solvent", value=6.3, format="%.6g")
    scale = st.number_input("Scale", min_value=0.0, value=1.0, format="%.6g")
    background = st.number_input("Background", min_value=0.0, value=1e-4, format="%.6g")

    st.header("Contour Length (L)")
    l_mean = st.number_input("L mean", min_value=1e-9, value=1000.0, format="%.6g")
    l_dist = st.selectbox(
        "L distribution",
        ["Gaussian", "Lognormal", "Schulz (Gamma)", "Triangular", "Uniform", "Boltzmann"],
        index=0,
    )
    l_pd = st.slider("L PD (sigma/mu)", min_value=0.0, max_value=20.0, value=0.10, step=0.01)
    l_bins = st.number_input("L bins", min_value=1, max_value=400, value=25)

    st.header("Kuhn Length (b)")
    b_mean = st.number_input("b mean", min_value=1e-9, value=100.0, format="%.6g")
    b_dist = st.selectbox(
        "b distribution",
        ["Gaussian", "Lognormal", "Schulz (Gamma)", "Triangular", "Uniform", "Boltzmann"],
        index=0,
    )
    b_pd = st.slider("b PD (sigma/mu)", min_value=0.0, max_value=20.0, value=0.10, step=0.01)
    b_bins = st.number_input("b bins", min_value=1, max_value=400, value=25)


if q_max <= q_min:
    st.error("q_max must be greater than q_min.")
    st.stop()

q = np.geomspace(q_min, q_max, int(q_points)) if q_log else np.linspace(q_min, q_max, int(q_points))

l_grid, l_pdf, l_w = build_distribution(DistSpec(name=l_dist, mean=l_mean, pd_index=l_pd, n_bins=int(l_bins)))
b_grid, b_pdf, b_w = build_distribution(DistSpec(name=b_dist, mean=b_mean, pd_index=b_pd, n_bins=int(b_bins)))

evaluator = SasmodelsEvaluator(q)

base = {
    "scale": float(scale),
    "radius": float(radius),
    "sld": float(sld),
    "sld_solvent": float(sld_solvent),
    "background": float(background),
    "length": float(l_mean),
    "kuhn_length": float(b_mean),
}

try:
    i_mean = compute_intensity(evaluator, base)
except Exception as exc:
    st.error(f"Failed to evaluate sasmodels flexible_cylinder for mean parameters: {exc}")
    st.stop()

poly_active = (l_pd > 0 and l_bins > 1) or (b_pd > 0 and b_bins > 1)

if poly_active:
    try:
        i_smeared = weighted_2d_polydisperse_iq(evaluator, q, base, l_grid, l_w, b_grid, b_w)
    except Exception as exc:
        st.error(f"Failed during 2D polydispersity integration: {exc}")
        st.stop()
else:
    i_smeared = i_mean.copy()

st.caption(f"sasmodels evaluation engine: {evaluator.engine_label}")

plot_mode = st.radio(
    "Plot Representation",
    ["Log-Log", "Lin-Lin", "Guinier", "Kratky", "Porod"],
    horizontal=True,
)

zoom_fit = False
y_label = "I(q)"
x_label = "q"
x_data = q.copy()
y_mean = i_mean.copy()
y_smeared = i_smeared.copy()

if plot_mode == "Guinier":
    x_data = q * q
    x_label = "q^2"
    y_label = "ln(I)"
    y_mean = np.log(np.clip(i_mean, 1e-300, None))
    y_smeared = np.log(np.clip(i_smeared, 1e-300, None))
elif plot_mode == "Kratky":
    y_label = "q^2 I(q)"
    y_mean = q * q * i_mean
    y_smeared = q * q * i_smeared
elif plot_mode == "Porod":
    y_label = "q^4 I(q)"
    q4 = q**4
    y_mean = q4 * i_mean
    y_smeared = q4 * i_smeared

fig = go.Figure()

if poly_active:
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_mean,
            mode="lines",
            name="Mean (monodisperse)",
            line=dict(dash="dash", width=2),
        )
    )

fig.add_trace(
    go.Scatter(
        x=x_data,
        y=y_smeared,
        mode="lines",
        name="Smeared" if poly_active else "Mean",
        line=dict(width=3),
    )
)

fig.update_layout(
    xaxis_title=x_label,
    yaxis_title=y_label,
    height=500,
    legend=dict(orientation="h", y=1.04),
)

if plot_mode == "Log-Log":
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
elif plot_mode == "Lin-Lin":
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="linear")

if plot_mode == "Guinier":
    st.subheader("Guinier Analysis")
    rg_theory = benoit_doty_rg(l_mean, b_mean)
    fit_mask = (q * rg_theory < 1.3) & np.isfinite(y_smeared)

    if np.count_nonzero(fit_mask) >= 2:
        fit_x = x_data[fit_mask]
        fit_y = y_smeared[fit_mask]
        fit = linregress(fit_x, fit_y)

        slope = fit.slope
        intercept = fit.intercept
        rg_app = np.sqrt(max(-3.0 * slope, 0.0))
        i0_app = float(np.exp(intercept))

        fit_line = intercept + slope * fit_x
        fig.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_line,
                mode="lines",
                name="Guinier fit (qRg<1.3)",
                line=dict(width=2),
            )
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Theoretical Rg", f"{rg_theory:.6g}")
        c2.metric("Apparent Rg", f"{rg_app:.6g}")
        c3.metric("Apparent I(0)", f"{i0_app:.6g}")
        c4.metric("R^2", f"{fit.rvalue**2:.6f}")

        st.caption("Fit performed on ln(I) vs q^2 where q*Rg(theory) < 1.3")

        zoom_fit = st.button("Zoom to Fit Region")
        if zoom_fit:
            x_max_fit = float(np.max(fit_x))
            y_min_fit = float(np.min(fit_y))
            y_max_fit = float(np.max(fit_y))
            pad_y = 0.08 * max(1e-12, y_max_fit - y_min_fit)
            fig.update_xaxes(range=[0.0, x_max_fit])
            fig.update_yaxes(range=[y_min_fit - pad_y, y_max_fit + pad_y])
    else:
        st.warning("Not enough points satisfy q*Rg < 1.3 for Guinier linear regression.")

st.plotly_chart(fig, width='stretch')

st.subheader("Distribution PDFs")
col_l, col_b = st.columns(2)

with col_l:
    f_l = go.Figure()
    f_l.add_trace(go.Scatter(x=l_grid, y=l_pdf, mode="lines+markers", name="L PDF"))
    f_l.update_layout(height=300, xaxis_title="L", yaxis_title="Probability")
    st.plotly_chart(f_l, width='stretch')

with col_b:
    f_b = go.Figure()
    f_b.add_trace(go.Scatter(x=b_grid, y=b_pdf, mode="lines+markers", name="b PDF"))
    f_b.update_layout(height=300, xaxis_title="b", yaxis_title="Probability")
    st.plotly_chart(f_b, width='stretch')

metadata = {
    "model": "sasmodels:flexible_cylinder",
    "q_min": q_min,
    "q_max": q_max,
    "n_q": int(q_points),
    "q_spacing": "log" if q_log else "linear",
    "radius": radius,
    "sld": sld,
    "sld_solvent": sld_solvent,
    "scale": scale,
    "background": background,
    "L_mean": l_mean,
    "L_distribution": l_dist,
    "L_PD": l_pd,
    "L_bins": int(l_bins),
    "b_mean": b_mean,
    "b_distribution": b_dist,
    "b_PD": b_pd,
    "b_bins": int(b_bins),
}

if plot_mode == "Guinier":
    metadata["Rg_theory"] = benoit_doty_rg(l_mean, b_mean)

export_df = pd.DataFrame({"q": q, "I_mean": i_mean, "I_smeared": i_smeared})
header = format_metadata(metadata)
buf = io.StringIO()
buf.write(header)
export_df.to_csv(buf, index=False)

st.download_button(
    label="Download CSV",
    data=buf.getvalue(),
    file_name="saxs_wlc_simulation.csv",
    mime="text/csv",
)

params_txt = format_parameter_dump(metadata)
st.download_button(
    label="Download Params TXT",
    data=params_txt,
    file_name="saxs_wlc_parameters.txt",
    mime="text/plain",
)

dump_path = st.text_input("Parameter dump path", value="saxs_wlc_parameters.txt")
if st.button("Dump Params TXT to Disk"):
    try:
        dump_parameters_txt(metadata, dump_path)
        st.success(f"Parameter dump saved to: {dump_path}")
    except Exception as exc:
        st.error(f"Failed to write parameter dump: {exc}")
