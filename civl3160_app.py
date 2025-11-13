"""
CIVL3160 – Transportation Engineering
Streamlit web app – v3

Modules:
- Overview (what each topic/assignment is about)
- PHF & sub-interval flows (Topic 04 / Assignment 1)
- Greenshields capacity & plots (Topic 06 / Assignment 3)
- Basic freeway LOS (Topics 07–08 / Assignment 3)
- Speed analysis (Topic 05 / Assignment 2)
- Safety – crash rate & EPDO (Topics 09–11 / Assignment 4)
- Safety – countermeasures CRF/CMF (Topic 12 / Assignment 4)

This app is for learning & visualization. For graded work or design,
always double-check with the Highway Capacity Manual and your notes.
"""

import math
import statistics
from typing import Sequence, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# CONSTANTS & LOW-LEVEL HELPERS
# ---------------------------------------------------------------------------

SECONDS_PER_HOUR = 3600.0
METERS_PER_KM = 1000.0
PI = math.pi


def _safe_div(num: float, den: float) -> float:
    """Safe division: returns 0 if denominator is zero."""
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# TRAFFIC FORMULAS
# ---------------------------------------------------------------------------

# --- Volume & PHF -----------------------------------------------------------

def hourly_flow_from_subhour_flow(q_sub: float, interval_minutes: float) -> float:
    """
    Convert sub-hour flow to hourly:
        q_60 = q_sub * (60 / Δt)
    """
    factor = _safe_div(60.0, interval_minutes)
    return q_sub * factor


def peak_hour_factor_general(total_hour_volume: float, peak_sub_volume: float, n_sub: int) -> float:
    """
    Generalized PHF for N equal sub-intervals:

        PHF = V / (N * q_sub,max)

    where:
        V           = total hourly volume [veh/h]
        N           = number of sub-intervals
        q_sub,max   = maximum sub-interval flow [veh/sub-interval]
    """
    return _safe_div(total_hour_volume, n_sub * peak_sub_volume)


def ddhv(aadt_value: float, K: float, D: float) -> float:
    """
    Directional Design Hour Volume (DDHV):

        DDHV = AADT * K * D
    """
    return aadt_value * K * D


# --- Speed & normal distribution -------------------------------------------

def time_mean_speed_disaggregate(speeds: Sequence[float]) -> float:
    """Time mean speed v_T = (1/N) Σ v_i."""
    if not speeds:
        return 0.0
    return statistics.mean(speeds)


def space_mean_speed_from_speeds(speeds: Sequence[float]) -> float:
    """
    Space mean speed (harmonic mean):

        v_s = N / Σ(1 / v_i)
    """
    if not speeds:
        return 0.0
    inv_sum = sum(_safe_div(1.0, v) for v in speeds if v > 0)
    N = len([v for v in speeds if v > 0])
    return _safe_div(float(N), inv_sum)


def percentile_speed_disaggregate(speeds: Sequence[float], p: float) -> float:
    """
    Percentile speed (Topic 05) using rank interpolation.
    """
    if not speeds:
        return 0.0
    data = sorted(speeds)
    N = len(data)
    if N == 1:
        return data[0]

    Rank = 1.0 + (p * (N - 1) / 100.0)
    r = int(math.floor(Rank))
    d = Rank - r

    idx_r = max(0, min(r - 1, N - 1))
    if idx_r == N - 1:
        return data[idx_r]

    idx_r1 = idx_r + 1
    return data[idx_r] + d * (data[idx_r1] - data[idx_r])


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Normal pdf: f(x) = 1/(σ√(2π)) exp(-(x-μ)²/(2σ²)).
    """
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * PI))


# --- Greenshields model -----------------------------------------------------

def greenshields_capacity(vf: float, kj: float) -> Tuple[float, float, float]:
    """
    Greenshields linear speed–density model:

        v_s(k) = v_f (1 - k/k_j)

    Capacity occurs at:
        k_m = k_j / 2
        v_m = v_f / 2
        q_m = v_f k_j / 4
    """
    k_m = kj / 2.0
    v_m = vf / 2.0
    q_m = (vf * kj) / 4.0
    return q_m, v_m, k_m


# --- Basic freeway LOS (HCM-style, simplified) -----------------------------

def heavy_vehicle_adjustment_factor(PT: float, ET: float, PR: float, ER: float) -> float:
    """
    Heavy vehicle adjustment factor:

        f_HV = 1 / [1 + PT(ET - 1) + PR(ER - 1)]
    """
    denominator = 1.0 + PT * (ET - 1.0) + PR * (ER - 1.0)
    return _safe_div(1.0, denominator)


def passenger_car_equivalent_flow_per_lane(
    V: float,
    PHF: float,
    N_lanes: int,
    fHV: float,
    fP: float = 1.0,
) -> float:
    """
    Passenger-car equivalent flow per lane:

        v_p = V / (PHF * N * f_HV * f_P)
    """
    den = PHF * N_lanes * fHV * fP
    return _safe_div(V, den)


def density_basic_freeway(vp: float, S_mi_per_h: float) -> float:
    """
    Basic freeway density:

        D = v_p / S   [pc/mi/ln]
    """
    return _safe_div(vp, S_mi_per_h)


def los_from_density(
    D: float,
    thresholds: Dict[str, Tuple[float, float]] = None
) -> str:
    """
    LOS classification by density ranges.
    Default thresholds are typical HCM-like values (A–F).
    """
    default_thresholds = {
        "A": (0.0, 11.0),
        "B": (11.0, 18.0),
        "C": (18.0, 26.0),
        "D": (26.0, 35.0),
        "E": (35.0, 45.0),
        "F": (45.0, float("inf")),
    }
    if thresholds is None:
        thresholds = default_thresholds

    for los, (d_min, d_max) in thresholds.items():
        if d_min <= D < d_max:
            return los
    return "F"


# ---------------------------------------------------------------------------
# ROAD SAFETY FORMULAS (Topics 09–12)
# ---------------------------------------------------------------------------

def tmev_intersection(major_aadt: float, minor_aadt: float, years: float) -> float:
    """
    Total Million Entering Vehicles (TMEV) – intersection:

        TEV  = AADT_major + AADT_minor
        TMEV = (TEV * Y * 365) / 10^6
    """
    tev = major_aadt + minor_aadt
    return tev * years * 365.0 / 1_000_000.0


def mvk_segment(aadt: float, length_km: float, years: float) -> float:
    """
    Million Vehicle-kilometres (MVK) – segment:

        MVK = (AADT * L * Y * 365) / 10^6
    """
    return aadt * length_km * years * 365.0 / 1_000_000.0


def crash_rate(K_total: float, exposure: float) -> float:
    """
    Crash rate:

        Crash Rate = K_total / exposure
    """
    return _safe_div(K_total, exposure)


def epdo_score(KF: float, KI: float, KPDO: float, FW: float, IW: float, PW: float = 1.0) -> float:
    """
    EPDO score for one site:

        EPDO_i = F_W * K_F + I_W * K_I + P_W * K_PDO

    where weights F_W, I_W, P_W come from crash costs.
    """
    return FW * KF + IW * KI + PW * KPDO


def crf_from_cmf(cmf: float) -> float:
    """
    CRF (%) from CMF:

        CRF = (1 - CMF) * 100
    """
    return (1.0 - cmf) * 100.0


def cmf_from_crf(crf_percent: float) -> float:
    """
    CMF from CRF (%):

        CMF = 1 - CRF/100
    """
    return 1.0 - crf_percent / 100.0


def predicted_crashes_with_cmfs(N_base: float, cmfs: Sequence[float]) -> float:
    """
    N_pred = N_base * (CMF1 * CMF2 * ... * CMFn)
    """
    product = 1.0
    for c in cmfs:
        product *= c
    return N_base * product


# ---------------------------------------------------------------------------
# UI HELPERS – FORMULAS & INPUT PARSING
# ---------------------------------------------------------------------------

def formula_plain(latex: str, caption: str = ""):
    """
    Simple centered LaTeX with optional caption.
    IMPORTANT: pass only the formula, WITHOUT $$.
    """
    st.latex(latex)
    if caption:
        st.caption(caption)


FORMULA_BOX_CSS = """
<style>
.formula-box {
    background-color: #161925;
    padding: 0.75rem 1.0rem;
    border-radius: 0.6rem;
    border: 1px solid #2f3445;
    margin-top: 0.35rem;
    margin-bottom: 0.9rem;
}
</style>
"""


FORMULA_BOX_CSS = """
<style>
.formula-box {
    background-color: #161925;
    padding: 0.75rem 1.0rem;
    border-radius: 0.6rem;
    border: 1px solid #2f3445;
    margin-top: 0.35rem;
    margin-bottom: 0.9rem;
}
.formula-box .formula-inner {
    text-align: center;
}
.formula-caption {
    font-size: 0.8rem;
    color: #9aa0b5;
    margin-top: -0.35rem;
    margin-bottom: 0.5rem;
}
</style>
"""


def formula_box(latex: str, caption: str = ""):
    """
    A “card-style” formula, but still rendered with st.latex.
    IMPORTANT: pass only the formula, WITHOUT $$.
    """
    st.markdown(FORMULA_BOX_CSS, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(latex)
    st.markdown('</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def parse_speeds_input(text: str) -> List[float]:
    """
    Generic parser for comma / space / semicolon separated numbers.
    Works for speeds and lists of CMFs.
    """
    if not text.strip():
        return []
    text = text.replace(",", " ").replace(";", " ")
    parts = [p for p in text.split(" ") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# PAGE 0 – OVERVIEW
# ---------------------------------------------------------------------------

def page_overview():
    st.title("CIVL3160 – Transportation Engineering Toolbox")

    st.markdown(
        """
This app is organised to follow your course:

- **Topic 04 / Assignment 1 – Traffic Volume**  
  ADT/AADT ideas, design hour, PHF, DDHV.

- **Topic 05 / Assignment 2 – Speed**  
  Spot speed data, time/space mean, variance, percentiles, normal model.

- **Topic 06 / Assignment 3 – Traffic Flow Theory**  
  Fundamental diagram, Greenshields, capacity.

- **Topics 07–08 / Assignment 3 – Level of Service**  
  Basic freeway sections, density, heavy vehicles, LOS A–F.

- **Topics 09–11 / Assignment 4 – Road Safety & Network Screening**  
  Crash rate, exposure (TMEV, MVK), EPDO scoring.

- **Topic 12 / Assignment 4 – Countermeasures**  
  Crash reduction factors (CRF), crash modification factors (CMF), predicting
  expected crashes after improvements.
        """
    )

    st.markdown(
        """
Use the sidebar to jump to each module.  
The idea is: **same formulas as in lectures, but interactive.**  
Type numbers → see outputs, plots, and LOS/crash metrics instantly.
        """
    )


# ---------------------------------------------------------------------------
# PAGE 1 – PHF & SUB-INTERVAL FLOWS
# ---------------------------------------------------------------------------

def page_phf():
    st.header("PHF & Sub-Interval Flow (Assignment 1 / Topic 04)")

    formula_box(
        r"\text{PHF} = \dfrac{V}{N \cdot q_{\text{sub,max}}}",
        "Peak Hour Factor using total hourly volume V and maximum sub-interval flow.",
    )

    st.info(
        "This module corresponds to **Traffic Volume** analysis (ADT/AADT, DDHV, PHF) "
        "from Topic 04 and Assignment 1."
    )

    mode = st.radio(
        "How do you want to enter data?",
        ("Enter all sub-interval flows", "I only know total V and peak sub-interval flow"),
    )

    colN, colDt = st.columns(2)
    with colN:
        n = st.number_input("Number of sub-intervals N", min_value=1, max_value=24, value=4, step=1)
    with colDt:
        dt = st.number_input("Sub-interval length Δt (minutes)", min_value=1.0, max_value=60.0, value=15.0, step=1.0)

    if mode == "Enter all sub-interval flows":
        st.write("Enter each sub-interval flow (change N to add/remove).")
        flows: List[float] = []
        cols = st.columns(min(4, int(n)) or 1)
        for i in range(int(n)):
            col = cols[i % len(cols)]
            with col:
                flows.append(
                    st.number_input(
                        f"q{i + 1} (veh/{int(dt)}min)",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        key=f"q_{i}",
                    )
                )

        if st.button("Compute PHF", key="phf_btn1"):
            if not flows:
                st.error("No flows entered.")
                return

            V = sum(flows)
            q_peak = max(flows)
            q60_peak = hourly_flow_from_subhour_flow(q_peak, dt)
            phf_val = peak_hour_factor_general(V, q_peak, int(n))

            st.subheader("Results")
            st.write(f"Total hourly volume **V** = {V:.1f} veh/h")
            st.write(f"Peak sub-interval flow = {q_peak:.1f} veh/{int(dt)}min (= {q60_peak:.1f} veh/h)")
            st.write(f"PHF = V / (N · q_sub,max) = **{phf_val:.3f}**")

    else:
        V = st.number_input("Total hourly volume V (veh/h)", min_value=0.0, value=0.0, step=50.0)
        q_peak = st.number_input(
            f"Peak sub-interval flow q_sub,max (veh/{int(dt)}min)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

        if st.button("Compute PHF", key="phf_btn2"):
            q60_peak = hourly_flow_from_subhour_flow(q_peak, dt)
            phf_val = peak_hour_factor_general(V, q_peak, int(n))

            st.subheader("Results")
            st.write(f"Peak sub-interval flow = {q_peak:.1f} veh/{int(dt)}min (= {q60_peak:.1f} veh/h)")
            st.write(f"PHF = V / (N · q_sub,max) = **{phf_val:.3f}**")


# ---------------------------------------------------------------------------
# PAGE 2 – GREENSHIELDS
# ---------------------------------------------------------------------------

def page_greenshields():
    st.header("Greenshields Capacity & Diagrams (Assignment 3 / Topic 06)")

    formula_box(
        r"v_s(k) = v_f \left(1 - \dfrac{k}{k_j}\right)",
        "Greenshields linear speed–density model.",
    )
    formula_plain(
        r"k_m = \dfrac{k_j}{2}, \quad v_m = \dfrac{v_f}{2}, \quad q_m = \dfrac{v_f k_j}{4}",
        "Capacity (maximum flow) occurs at k = k_m.",
    )

    st.info(
        "This module connects to **Traffic Flow Theory** and the fundamental diagram from "
        "Topic 06 and the flow–density part of Assignment 3."
    )

    col1, col2 = st.columns(2)
    with col1:
        vf = st.number_input("Free-flow speed v_f (km/h)", min_value=1.0, value=100.0, step=1.0)
    with col2:
        kj = st.number_input("Jam density k_j (veh/km/lane)", min_value=1.0, value=200.0, step=5.0)

    if st.button("Compute & plot Greenshields"):
        qm, vm, km = greenshields_capacity(vf, kj)

        st.subheader("Capacity point")
        st.write(f"q_m = {qm:.1f} veh/h/lane")
        st.write(f"v_m = {vm:.1f} km/h")
        st.write(f"k_m = {km:.1f} veh/km/lane")

        k_vals = np.linspace(0, kj, 200)
        v_vals = vf * (1 - k_vals / kj)
        q_vals = vf * k_vals * (1 - k_vals / kj)

        fig1, ax1 = plt.subplots()
        ax1.plot(k_vals, v_vals)
        ax1.set_xlabel("k (veh/km/lane)")
        ax1.set_ylabel("v_s (km/h)")
        ax1.set_title("Speed–Density (Greenshields)")
        ax1.grid(True, alpha=0.3)

        fig2, ax2 = plt.subplots()
        ax2.plot(k_vals, q_vals)
        ax2.set_xlabel("k (veh/km/lane)")
        ax2.set_ylabel("q (veh/h/lane)")
        ax2.set_title("Flow–Density (Greenshields)")
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig1)
        st.pyplot(fig2)


# ---------------------------------------------------------------------------
# PAGE 3 – BASIC FREEWAY LOS
# ---------------------------------------------------------------------------

def page_basic_freeway_los():
    st.header("Basic Freeway LOS (Assignment 3 / Topics 07–08)")

    formula_box(
        r"f_{HV} = \dfrac{1}{1 + P_T(E_T-1) + P_R(E_R-1)}",
        "Heavy vehicle adjustment factor.",
    )
    formula_plain(
        r"v_p = \dfrac{V}{PHF \cdot N \cdot f_{HV} \cdot f_P}",
        "Passenger-car equivalent flow per lane.",
    )
    formula_plain(
        r"D = \dfrac{v_p}{S} \quad [pc/mi/ln]",
        "Density for basic freeway segments.",
    )

    st.info(
        "This mirrors the **basic freeway section LOS** procedure from Topics 07–08 "
        "and the LOS questions in Assignment 3."
    )

    mode = st.radio(
        "Input mode:",
        ("Compute D and LOS from volumes", "Just give me LOS from a known density D"),
    )

    if mode.startswith("Compute"):
        col1, col2 = st.columns(2)
        with col1:
            V = st.number_input("Design hour volume V (veh/h)", min_value=0.0, value=4000.0, step=100.0)
            PHF = st.number_input("Peak hour factor PHF", min_value=0.1, max_value=1.0, value=0.85, step=0.01)
            N = st.number_input("Number of lanes (one direction)", min_value=1, max_value=8, value=3, step=1)
        with col2:
            PT = st.number_input("Proportion trucks P_T (0–1)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
            ET = st.number_input("Truck PCE E_T", min_value=1.0, value=1.5, step=0.1)
            PR = st.number_input("Proportion RVs P_R (0–1)", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
            ER = st.number_input("RV PCE E_R", min_value=1.0, value=1.2, step=0.1)

        S = st.number_input("Analysis speed S (mi/h)", min_value=1.0, value=65.0, step=1.0)
        fP = st.number_input("Driver population factor f_P", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

        if st.button("Compute LOS"):
            fHV = heavy_vehicle_adjustment_factor(PT, ET, PR, ER)
            vp = passenger_car_equivalent_flow_per_lane(V, PHF, N, fHV, fP)
            D = density_basic_freeway(vp, S)
            los = los_from_density(D)

            st.subheader("Results")
            st.write(f"f_HV = {fHV:.3f}")
            st.write(f"v_p = {vp:.1f} pc/h/ln")
            st.write(f"D   = {D:.2f} pc/mi/ln")
            st.write(f"LOS ≈ **{los}**")
    else:
        D = st.number_input("Density D (pc/mi/ln)", min_value=0.0, value=20.0, step=1.0)
        if st.button("Get LOS from D"):
            los = los_from_density(D)
            st.write(f"LOS ≈ **{los}**")


# ---------------------------------------------------------------------------
# PAGE 4 – SPEED ANALYSIS
# ---------------------------------------------------------------------------

def page_speed_analysis():
    st.header("Speed Analysis & Normal Approximation (Assignment 2 / Topic 05)")

    formula_box(
        r"v_T = \dfrac{1}{N}\sum_{i=1}^{N} v_i, \quad "
        r"v_s = \dfrac{N}{\sum_{i=1}^{N} \dfrac{1}{v_i}}",
        "Time mean vs space mean speed.",
    )
    st.info(
        "This module matches **Speed analysis** (spot speeds, basic statistics, normal distribution, "
        "percentile speeds) from Topic 05 and Assignment 2."
    )

    default_speeds = "72, 68, 70, 75, 80, 66, 74, 71, 69, 73"
    speeds_text = st.text_area(
        "Spot speeds (km/h), separated by comma / space / semicolon:",
        value=default_speeds,
        height=110,
    )
    speeds = parse_speeds_input(speeds_text)

    if st.button("Analyze speeds"):
        if not speeds:
            st.error("No valid speeds provided.")
            return

        vT = time_mean_speed_disaggregate(speeds)
        vS = space_mean_speed_from_speeds(speeds)
        try:
            sigma = statistics.pstdev(speeds)
        except statistics.StatisticsError:
            sigma = 0.0
        v85 = percentile_speed_disaggregate(speeds, 85.0)

        st.subheader("Numeric results")
        st.write(f"Time mean speed v_T = {vT:.2f} km/h")
        st.write(f"Space mean speed v_s = {vS:.2f} km/h")
        st.write(f"85th percentile speed ≈ {v85:.2f} km/h")
        st.write(f"Std dev σ ≈ {sigma:.2f} km/h")

        if sigma > 0:
            xs = np.linspace(min(speeds), max(speeds), 200)
            ys = [normal_pdf(x, vT, sigma) for x in xs]

            fig, ax = plt.subplots()
            ax.hist(speeds, bins="auto", density=True, alpha=0.5, label="Histogram")
            ax.plot(xs, ys, linewidth=2.0, label="Normal approx.")
            ax.set_xlabel("Speed (km/h)")
            ax.set_ylabel("Probability density")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.subheader("Distribution")
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# PAGE 5 – SAFETY: CRASH RATE & EPDO
# ---------------------------------------------------------------------------

def page_safety_rates_epdo():
    st.header("Road Safety – Crash Rate & EPDO (Assignment 4 / Topics 09–11)")

    # Intersection crash rate
    formula_box(
        r"\text{Crash Rate}_{\text{int}} = \dfrac{K_{\text{TOT}}}{\text{TMEV}}",
        "Intersection crash rate (crashes per million entering vehicles).",
    )
    formula_plain(
        r"\text{TMEV} = \dfrac{\bigl(AADT_{\text{maj}} + AADT_{\text{min}}\bigr)\,Y\,365}{10^6}",
        "Total Million Entering Vehicles (TMEV).",
    )

    # Segment crash rate
    formula_box(
        r"\text{Crash Rate}_{\text{seg}} = \dfrac{K_{\text{TOT}}}{\text{MVK}}",
        "Segment crash rate (crashes per million vehicle-kilometres).",
    )
    formula_plain(
        r"\text{MVK} = \dfrac{AADT \, L \, Y \, 365}{10^6}",
        "Million Vehicle-Kilometres (MVK).",
    )

    # EPDO
    formula_plain(
        r"\text{EPDO}_i = F_W K_{F,i} + I_W K_{I,i} + P_W K_{\text{PDO},i}",
        "EPDO average crash frequency method for ranking sites.",
    )

    st.info(
        "This module covers **Road Safety – Network Screening** (crash rates, EPDO scoring) "
        "from Topics 09–11 and Assignment 4."
    )

    mode = st.radio(
        "Select calculation:",
        ("Intersection crash rate", "Segment crash rate", "EPDO score for a site"),
    )

    if mode == "Intersection crash rate":
        col1, col2 = st.columns(2)
        with col1:
            aadt_maj = st.number_input("Major road AADT (veh/day)", min_value=0.0, value=30000.0, step=1000.0)
            aadt_min = st.number_input("Minor road AADT (veh/day)", min_value=0.0, value=5000.0, step=500.0)
            years = st.number_input("Study period Y (years)", min_value=0.1, value=3.0, step=0.5)
        with col2:
            Ktot = st.number_input("Total crashes in study period K_TOT", min_value=0.0, value=30.0, step=1.0)

        if st.button("Compute intersection crash rate"):
            tmev = tmev_intersection(aadt_maj, aadt_min, years)
            rate = crash_rate(Ktot, tmev)

            st.subheader("Results")
            st.write(f"TMEV = {tmev:.3f} million entering vehicles")
            st.write(f"Crash rate = {rate:.3f} crashes / TMEV")

    elif mode == "Segment crash rate":
        col1, col2 = st.columns(2)
        with col1:
            aadt = st.number_input("Segment AADT (veh/day)", min_value=0.0, value=10000.0, step=1000.0)
            length = st.number_input("Segment length L (km)", min_value=0.0, value=1.0, step=0.1)
            years = st.number_input("Study period Y (years)", min_value=0.1, value=3.0, step=0.5)
        with col2:
            Ktot = st.number_input("Total crashes in study period K_TOT", min_value=0.0, value=25.0, step=1.0)

        if st.button("Compute segment crash rate"):
            mvk = mvk_segment(aadt, length, years)
            rate = crash_rate(Ktot, mvk)

            st.subheader("Results")
            st.write(f"MVK = {mvk:.3f} million vehicle-km")
            st.write(f"Crash rate = {rate:.3f} crashes / MVK")

    else:  # EPDO score
        st.write("EPDO weights (you can adjust based on your table):")
        col1, col2 = st.columns(2)
        with col1:
            FW = st.number_input("Fatal crash weight F_W", min_value=0.0, value=542.0, step=10.0)
            IW = st.number_input("Injury crash weight I_W", min_value=0.0, value=11.0, step=1.0)
            PW = st.number_input("PDO crash weight P_W", min_value=0.0, value=1.0, step=0.5)
        with col2:
            KF = st.number_input("Number of fatal crashes K_F", min_value=0.0, value=0.0, step=1.0)
            KI = st.number_input("Number of injury crashes K_I", min_value=0.0, value=10.0, step=1.0)
            KPDO = st.number_input("Number of PDO crashes K_PDO", min_value=0.0, value=20.0, step=1.0)

        if st.button("Compute EPDO score"):
            score = epdo_score(KF, KI, KPDO, FW, IW, PW)
            st.subheader("Result")
            st.write(f"EPDO score for this site = **{score:.2f}**")

# ---------------------------------------------------------------------------
# PAGE 6 – SAFETY: COUNTERMEASURES CRF / CMF
# ---------------------------------------------------------------------------

def page_safety_crf_cmf():
    st.header("Road Safety – Countermeasures CRF/CMF (Assignment 4 / Topic 12)")

    formula_box(
        r"\text{CRF} = (1 - \text{CMF}) \times 100",
        "Crash Reduction Factor (percent reduction) from a Crash Modification Factor.",
    )
    formula_plain(
        r"N_{\text{pred}} = N_{\text{base}} \cdot (CMF_1 \cdot CMF_2 \cdots CMF_n)",
        "Expected average crashes with countermeasures applied.",
    )

    st.info(
        "This module corresponds to **Select Countermeasures** (using CRF/CMF) from Topic 12 and "
        "the safety design part of Assignment 4."
    )

    st.subheader("1) Convert between CRF and CMF")

    col1, col2 = st.columns(2)
    with col1:
        cmf_in = st.number_input("CMF (e.g., 0.88)", min_value=0.0, max_value=5.0, value=0.88, step=0.01)
        if st.button("CMF → CRF"):
            crf_val = crf_from_cmf(cmf_in)
            st.write(f"CRF = **{crf_val:.1f}%** crash reduction")
    with col2:
        crf_in = st.number_input("CRF (%) (e.g., 20)", min_value=-100.0, max_value=100.0, value=20.0, step=1.0)
        if st.button("CRF → CMF"):
            cmf_val = cmf_from_crf(crf_in)
            st.write(f"CMF = **{cmf_val:.3f}**")

    st.subheader("2) Predict crashes with multiple CMFs")

    base = st.number_input("Base expected crashes N_base (crashes/year)", min_value=0.0, value=25.98, step=0.5)
    cmf_text = st.text_area(
        "List of CMFs to apply (comma/space/semicolon separated):",
        value="0.80 0.91 0.62",  # example: 20%, 9%, 38% reduction
        height=80,
    )
    cmf_list = parse_speeds_input(cmf_text)

    if st.button("Compute N_pred and crash reduction"):
        if not cmf_list:
            st.error("Please enter at least one CMF.")
            return
        N_pred = predicted_crashes_with_cmfs(base, cmf_list)
        reduction = base - N_pred
        percent_red = _safe_div(reduction, base) * 100.0 if base > 0 else 0.0

        st.subheader("Results")
        st.write(f"N_base = {base:.2f} crashes/year")
        st.write(f"N_pred = {N_pred:.2f} crashes/year (after countermeasures)")
        st.write(f"Absolute reduction = {reduction:.2f} crashes/year")
        st.write(f"Percent reduction ≈ {percent_red:.1f}%")

    st.caption(
        "Remember: CMFs are usually obtained from sources like the HSM or FHWA CMF Clearinghouse. "
        "Always check that the CMFs you use match your crash type and site type."
    )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="CIVL3160 Traffic Toolbox", layout="wide")

    st.sidebar.title("CIVL3160 Toolbox")
    page = st.sidebar.radio(
        "Select module:",
        (
            "Overview",
            "PHF & sub-interval flows",
            "Greenshields capacity & plots",
            "Basic freeway LOS",
            "Speed analysis",
            "Safety – crash rate & EPDO",
            "Safety – CRF / CMF",
        ),
    )

    st.sidebar.info(
        "Use this app to explore formulas **interactively**.\n\n"
        "Good for understanding, checking homework ideas, and building intuition.\n\n"
        "For graded work or real design, always confirm with your lecture notes and the Highway Capacity Manual."
    )

    if page == "Overview":
        page_overview()
    elif page == "PHF & sub-interval flows":
        page_phf()
    elif page == "Greenshields capacity & plots":
        page_greenshields()
    elif page == "Basic freeway LOS":
        page_basic_freeway_los()
    elif page == "Speed analysis":
        page_speed_analysis()
    elif page == "Safety – crash rate & EPDO":
        page_safety_rates_epdo()
    elif page == "Safety – CRF / CMF":
        page_safety_crf_cmf()


if __name__ == "__main__":
    main()
