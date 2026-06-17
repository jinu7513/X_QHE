from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from he_hhl_solver import run_analysis


st.set_page_config(
    page_title="CKKS + HHL 보안 회귀 분석",
    page_icon="Q",
    layout="wide",
)


LINE_LABELS = {
    "classical": "Classical Regression",
    "hhl": "Quantum HHL",
    "he_hhl": "CKKS + HHL",
}

SERIES_LABELS = ["Actual Data", "Classical Regression", "Quantum HHL", "CKKS + HHL"]
SERIES_COLORS = ["#111827", "#0ea5e9", "#7c3aed", "#d97706"]


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.22);
        }
        .app-hero {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 1.35rem 1.5rem 1.25rem;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 58%, #eef6ff 100%);
            margin-bottom: 1rem;
        }
        .app-eyebrow {
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        .app-title {
            color: #0f172a;
            font-size: 2.1rem;
            line-height: 1.18;
            font-weight: 760;
            letter-spacing: 0;
            margin: 0;
        }
        .app-subtitle {
            color: #334155;
            font-size: 1rem;
            line-height: 1.65;
            max-width: 920px;
            margin-top: 0.55rem;
            margin-bottom: 0.95rem;
        }
        .pill-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 999px;
            padding: 0.26rem 0.62rem;
            background: rgba(255, 255, 255, 0.72);
            color: #0f172a;
            font-size: 0.82rem;
            font-weight: 650;
            white-space: nowrap;
        }
        .section-label {
            margin-top: 0.45rem;
            margin-bottom: 0.35rem;
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 760;
        }
        .flow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin-bottom: 1rem;
        }
        .flow-item {
            min-height: 112px;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 0.82rem 0.9rem;
            background: #ffffff;
        }
        .flow-index {
            color: #0369a1;
            font-size: 0.78rem;
            font-weight: 780;
            margin-bottom: 0.35rem;
        }
        .flow-title {
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }
        .flow-copy {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.48;
        }
        .result-note {
            border-left: 4px solid #0ea5e9;
            padding: 0.72rem 0.9rem;
            background: #f8fafc;
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }
        @media (max-width: 900px) {
            .flow-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .app-title {
                font-size: 1.55rem;
            }
        }
        @media (max-width: 560px) {
            .flow-grid {
                grid-template-columns: 1fr;
            }
            .app-hero {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-eyebrow">보안 데이터 분석 프로젝트</div>
            <h1 class="app-title">CKKS 동형암호와 HHL로 보는 보안 회귀 분석</h1>
            <div class="app-subtitle">
                스프레드시트 데이터를 입력하면 일반 선형회귀, Qiskit 기반 HHL, CKKS로 암호화된
                집계값을 사용한 HHL 결과를 한 화면에서 비교합니다. 연구 확장 파트에서는
                amplitude encoding과 QOTP 기반 QHE handoff가 데이터 상태를 보존하는지도 확인합니다.
            </div>
            <div class="pill-row">
                <span class="pill">CKKS encrypted aggregation</span>
                <span class="pill">Qiskit HHL solver</span>
                <span class="pill">QHE handoff verification</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_overview() -> None:
    st.markdown('<div class="section-label">분석 흐름</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-grid">
            <div class="flow-item">
                <div class="flow-index">STEP 1</div>
                <div class="flow-title">데이터 선택</div>
                <div class="flow-copy">CSV 또는 Excel에서 x 컬럼과 분석할 y 컬럼을 고릅니다. 날짜 x값은 일 단위 숫자로 변환됩니다.</div>
            </div>
            <div class="flow-item">
                <div class="flow-index">STEP 2</div>
                <div class="flow-title">CKKS 집계</div>
                <div class="flow-copy">y, z, z*y, z^2 같은 회귀용 통계량을 CKKS 암호문 상태에서 계산합니다.</div>
            </div>
            <div class="flow-item">
                <div class="flow-index">STEP 3</div>
                <div class="flow-title">HHL 비교</div>
                <div class="flow-copy">복호화된 집계값으로 A beta = b를 만들고 HHL 방식의 선형시스템 풀이 결과를 비교합니다.</div>
            </div>
            <div class="flow-item">
                <div class="flow-index">STEP 4</div>
                <div class="flow-title">QHE handoff</div>
                <div class="flow-copy">전체 y값을 amplitude encoding한 뒤 QOTP/Clifford 검증을 거쳐 HHL로 넘기는 연구용 흐름을 보여줍니다.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value, precision: int = 6) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def read_spreadsheet(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    contents = uploaded_file.getvalue()
    if name.endswith((".csv", ".txt")):
        return pd.read_csv(io.BytesIO(contents))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contents))
    raise ValueError("CSV, TXT, XLSX, XLS 파일만 지원합니다.")


def parse_x_axis(series: pd.Series) -> tuple[np.ndarray, list[str], str]:
    labels = series.astype(str).tolist()

    if pd.api.types.is_datetime64_any_dtype(series):
        dates = pd.to_datetime(series, errors="coerce")
        finite_dates = dates.dropna()
        if len(finite_dates) < 2:
            raise ValueError("X 컬럼에는 유효한 숫자 또는 날짜 값이 최소 2개 이상 필요합니다.")
        origin = finite_dates.min()
        values = ((dates - origin).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
        return values, labels, f"{series.name} (days from {origin.date()})"

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= 2:
        return numeric.to_numpy(dtype=float), labels, str(series.name)

    dates = pd.to_datetime(series, errors="coerce")
    if dates.notna().sum() >= 2:
        origin = dates.dropna().min()
        values = ((dates - origin).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
        return values, labels, f"{series.name} (days from {origin.date()})"

    raise ValueError("X 컬럼에는 유효한 숫자 또는 날짜 값이 최소 2개 이상 필요합니다.")


def finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_chart_frame(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_labels: list[str],
    result: dict,
) -> pd.DataFrame:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return pd.DataFrame(columns=["x", "x_label", "y", "series", "kind"])

    sorted_indices = indices[np.argsort(x_values[indices])]
    rows: list[dict[str, object]] = []

    for i in sorted_indices:
        rows.append(
            {
                "x": float(x_values[i]),
                "x_label": x_labels[i] if i < len(x_labels) else str(i),
                "y": float(y_values[i]),
                "series": "Actual Data",
                "kind": "actual",
            }
        )

    for key, label in LINE_LABELS.items():
        method = result.get(key, {})
        intercept = finite_float(method.get("intercept"))
        slope = finite_float(method.get("slope"))
        if intercept is None or slope is None:
            continue
        for i in sorted_indices:
            rows.append(
                {
                    "x": float(x_values[i]),
                    "x_label": x_labels[i] if i < len(x_labels) else str(i),
                    "y": float(intercept + slope * x_values[i]),
                    "series": label,
                    "kind": "line",
                }
            )

    return pd.DataFrame(rows)


def render_chart(chart_df: pd.DataFrame, x_title: str, y_title: str) -> None:
    if chart_df.empty:
        st.warning("그래프로 표시할 유효한 데이터가 없습니다.")
        return

    spec = {
        "height": 380,
        "layer": [
            {
                "transform": [{"filter": "datum.kind == 'actual'"}],
                "mark": {"type": "point", "filled": True, "size": 70, "opacity": 0.9},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "title": x_title},
                    "y": {"field": "y", "type": "quantitative", "title": y_title},
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {
                            "domain": SERIES_LABELS,
                            "range": SERIES_COLORS,
                        },
                    },
                    "tooltip": [
                        {"field": "x_label", "type": "nominal", "title": "x"},
                        {"field": "y", "type": "quantitative", "title": "y", "format": ".4f"},
                        {"field": "series", "type": "nominal", "title": "series"},
                    ],
                },
            },
            {
                "transform": [{"filter": "datum.kind == 'line'"}],
                "mark": {"type": "line", "strokeWidth": 3},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "title": x_title},
                    "y": {"field": "y", "type": "quantitative", "title": y_title},
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {
                            "domain": SERIES_LABELS,
                            "range": SERIES_COLORS,
                        },
                    },
                    "tooltip": [
                        {"field": "x_label", "type": "nominal", "title": "x"},
                        {"field": "y", "type": "quantitative", "title": "predicted y", "format": ".4f"},
                        {"field": "series", "type": "nominal", "title": "series"},
                    ],
                },
            },
        ],
        "config": {
            "background": "transparent",
            "axis": {"labelColor": "#94a3b8", "titleColor": "#f8fafc", "gridColor": "#334155"},
            "legend": {"labelColor": "#f8fafc", "titleColor": "#f8fafc"},
            "view": {"stroke": "transparent"},
        },
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_metrics(title: str, data: dict, include_stats: bool = False) -> None:
    st.subheader(title)
    if not data:
        st.info("결과가 없습니다.")
        return

    metric_labels = []
    if include_stats:
        metric_labels.extend([("Mean (y)", data.get("mean_y")), ("Variance (y)", data.get("var_y"))])
    metric_labels.extend(
        [
            ("Intercept", data.get("intercept")),
            ("Slope", data.get("slope")),
            ("R2", data.get("r2")),
        ]
    )

    cols = st.columns(len(metric_labels))
    for col, (label, value) in zip(cols, metric_labels):
        col.metric(label, format_number(value))


def render_encrypted_preview(preview: dict) -> None:
    if not preview:
        return

    with st.expander("CKKS 암호문 미리보기", expanded=False):
        for title, items in [
            ("입력 벡터", preview.get("vectors", [])),
            ("암호화된 집계값", preview.get("aggregates", [])),
        ]:
            if not items:
                continue
            st.markdown(f"**{title}**")
            for item in items:
                st.caption(
                    f"{item.get('name')} | {item.get('scheme')} | "
                    f"{item.get('value_count')} value(s) | {item.get('ciphertext_bytes')} bytes | "
                    f"sha256 {str(item.get('sha256', ''))[:12]}"
                )
                st.code(f"{item.get('base64_preview', '')}{'...' if item.get('truncated') else ''}")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.to_numeric(df[col], errors="coerce").notna().any()]


def render_dataset_summary(df: pd.DataFrame, x_choice: str, targets: list[str]) -> None:
    st.markdown('<div class="section-label">데이터 설정</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", f"{len(df):,}")
    c2.metric("컬럼 수", f"{len(df.columns):,}")
    c3.metric("x 컬럼", "행 번호" if x_choice == "Use row index" else str(x_choice))
    c4.metric("y 컬럼", f"{len(targets):,}개")

    selected_targets = ", ".join(map(str, targets)) if targets else "선택 전"
    st.markdown(
        f"""
        <div class="result-note">
            현재 선택된 분석 대상은 <b>{selected_targets}</b>입니다. 아래 그래프에서 실제 데이터 점과
            각 방법의 회귀 직선을 함께 비교합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("Amplitude encoding requires at least one value.")
    return 1 << int(np.ceil(np.log2(value)))


def prepare_amplitude_payload(values: np.ndarray) -> dict:
    y = np.asarray(values, dtype=float)
    if y.size < 2:
        raise ValueError("QHE handoff requires at least two finite target values.")

    mean = float(np.mean(y))
    centered = y - mean
    norm = float(np.linalg.norm(centered))
    if norm < 1.0e-12:
        raise ValueError("Selected target column is constant after centering; amplitude encoding is undefined.")

    padded_size = next_power_of_two(len(centered))
    n_qubits = int(np.log2(padded_size))
    padded = np.pad(centered, (0, padded_size - len(centered)))
    amplitudes = padded / norm

    return {
        "amplitudes": amplitudes.astype(complex),
        "mean": mean,
        "norm": norm,
        "original_length": len(centered),
        "padded_size": padded_size,
        "padding": padded_size - len(centered),
        "n_qubits": n_qubits,
    }


def apply_qotp(circuit: QuantumCircuit, a_key: np.ndarray, b_key: np.ndarray, inverse: bool = False) -> None:
    if inverse:
        for qubit, bit in enumerate(a_key):
            if int(bit):
                circuit.x(qubit)
        for qubit, bit in enumerate(b_key):
            if int(bit):
                circuit.z(qubit)
        return

    for qubit, bit in enumerate(b_key):
        if int(bit):
            circuit.z(qubit)
    for qubit, bit in enumerate(a_key):
        if int(bit):
            circuit.x(qubit)


def update_h_key(a_key: np.ndarray, b_key: np.ndarray, qubit: int) -> None:
    a_key[qubit], b_key[qubit] = int(b_key[qubit]), int(a_key[qubit])


def update_cx_key(a_key: np.ndarray, b_key: np.ndarray, control: int, target: int) -> None:
    a_key[target] = int(a_key[target]) ^ int(a_key[control])
    b_key[control] = int(b_key[control]) ^ int(b_key[target])


def build_identity_clifford_eval(n_qubits: int, a_key: np.ndarray, b_key: np.ndarray) -> tuple[QuantumCircuit, list[str]]:
    circuit = QuantumCircuit(n_qubits, name="qhe_identity_clifford_eval")
    operations: list[str] = []

    circuit.h(0)
    update_h_key(a_key, b_key, 0)
    operations.append("H(q0)")

    if n_qubits >= 2:
        circuit.cx(0, 1)
        update_cx_key(a_key, b_key, 0, 1)
        operations.append("CX(q0, q1)")

        circuit.cx(0, 1)
        update_cx_key(a_key, b_key, 0, 1)
        operations.append("CX(q0, q1)")

    circuit.h(0)
    update_h_key(a_key, b_key, 0)
    operations.append("H(q0)")
    return circuit, operations


def run_qhe_hhl_handoff(x_values: np.ndarray, y_values: np.ndarray, seed: int = 2026) -> dict:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = np.asarray(x_values[mask], dtype=float)
    y = np.asarray(y_values[mask], dtype=float)

    payload = prepare_amplitude_payload(y)
    n_qubits = int(payload["n_qubits"])
    amplitudes = payload["amplitudes"]
    initial_state = Statevector(amplitudes)

    rng = np.random.default_rng(seed)
    initial_a = rng.integers(0, 2, size=n_qubits, dtype=int)
    initial_b = rng.integers(0, 2, size=n_qubits, dtype=int)
    updated_a = initial_a.copy()
    updated_b = initial_b.copy()

    encrypt = QuantumCircuit(n_qubits, name="qotp_encrypt")
    apply_qotp(encrypt, initial_a, initial_b)

    evaluate, operations = build_identity_clifford_eval(n_qubits, updated_a, updated_b)

    decrypt = QuantumCircuit(n_qubits, name="qotp_decrypt")
    apply_qotp(decrypt, updated_a, updated_b, inverse=True)

    decrypted_state = initial_state.evolve(encrypt).evolve(evaluate).evolve(decrypt)
    fidelity = float(abs(np.vdot(initial_state.data, decrypted_state.data)) ** 2)

    recovered_centered = np.real_if_close(decrypted_state.data[: payload["original_length"]]).real * payload["norm"]
    recovered_y = recovered_centered + payload["mean"]
    reconstruction_error = float(np.max(np.abs(recovered_y - y)))

    handoff_result = run_analysis(x, recovered_y)

    return {
        "n_qubits": n_qubits,
        "padded_size": payload["padded_size"],
        "padding": payload["padding"],
        "mean": payload["mean"],
        "norm": payload["norm"],
        "initial_a": initial_a.tolist(),
        "initial_b": initial_b.tolist(),
        "updated_a": updated_a.tolist(),
        "updated_b": updated_b.tolist(),
        "operations": operations,
        "fidelity": fidelity,
        "reconstruction_error": reconstruction_error,
        "handoff_result": handoff_result,
    }


def render_qhe_hhl_handoff(x_values: np.ndarray, y_values: np.ndarray) -> None:
    with st.expander("QHE amplitude encoding handoff 검증", expanded=False):
        st.caption(
            "전체 y 컬럼을 scaling, padding, amplitude encoding한 뒤 Quantum One-Time Pad로 보호합니다. "
            "그 상태에서 identity-preserving Clifford 평가를 수행하고, 복호화 후 HHL 회귀 solver로 넘깁니다."
        )

        try:
            qhe = run_qhe_hhl_handoff(x_values, y_values)
        except Exception as exc:
            st.warning(f"QHE handoff 검증을 건너뛰었습니다: {exc}")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Amplitude qubits", qhe["n_qubits"])
        c2.metric("Encoded dimension", qhe["padded_size"])
        c3.metric("Zero padding", qhe["padding"])
        c4.metric("Fidelity", f"{qhe['fidelity']:.8f}")

        c5, c6 = st.columns(2)
        c5.metric("Max reconstruction error", f"{qhe['reconstruction_error']:.3e}")
        c6.metric("Scaling norm", f"{qhe['norm']:.6g}")

        st.markdown("**QOTP 키와 Clifford 평가**")
        st.code(
            "\n".join(
                [
                    f"initial a = {qhe['initial_a']}",
                    f"initial b = {qhe['initial_b']}",
                    f"evaluated gates = {' -> '.join(qhe['operations'])}",
                    f"updated a = {qhe['updated_a']}",
                    f"updated b = {qhe['updated_b']}",
                ]
            )
        )

        handoff = qhe["handoff_result"].get("hhl", {})
        st.markdown("**QHE 복호화 후 HHL 결과**")
        h1, h2, h3 = st.columns(3)
        h1.metric("Intercept", f"{float(handoff.get('intercept', float('nan'))):.6g}")
        h2.metric("Slope", f"{float(handoff.get('slope', float('nan'))):.6g}")
        h3.metric("R2", f"{float(handoff.get('r2', float('nan'))):.6g}")

        st.info(
            "이 영역은 제한적인 QHE handoff 검증입니다. HHL 전체를 암호화된 양자 상태에서 실행하는 것이 아니라, "
            "amplitude encoded 데이터가 QOTP/Clifford key update 과정을 거친 뒤 원래 상태로 복원되는지 확인하고 "
            "그 복원된 데이터를 HHL로 넘깁니다."
        )


inject_page_styles()
render_app_header()
render_pipeline_overview()

with st.sidebar:
    st.header("입력 데이터")
    uploaded_file = st.file_uploader("CSV / Excel 업로드", type=["csv", "txt", "xlsx", "xls"])
    use_sample = st.checkbox("샘플 데이터 사용", value=uploaded_file is None)

try:
    if uploaded_file is not None:
        df = read_spreadsheet(uploaded_file)
    elif use_sample:
        sample_path = ROOT_DIR / "data" / "example_quantum_chip_daily.csv"
        df = pd.read_csv(sample_path)
    else:
        st.info("왼쪽 사이드바에서 파일을 업로드하거나 샘플 데이터를 선택하세요.")
        st.stop()
except Exception as exc:
    st.error(f"파일을 읽을 수 없습니다: {exc}")
    st.stop()

if df.empty:
    st.error("데이터가 비어 있습니다.")
    st.stop()

with st.sidebar:
    st.header("분석 설정")
    x_options = ["Use row index"] + list(df.columns)
    default_x_index = x_options.index("date") if "date" in x_options else 0
    x_choice = st.selectbox("x 컬럼", x_options, index=default_x_index)

    y_options = numeric_columns(df)
    if x_choice in y_options:
        y_options = [col for col in y_options if col != x_choice]
    default_targets = y_options[:1]
    targets = st.multiselect("y 컬럼", y_options, default=default_targets)

render_dataset_summary(df, x_choice, targets)

with st.expander("데이터 미리보기", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

if not targets:
    st.warning("분석할 y 컬럼을 하나 이상 선택하세요.")
    st.stop()

try:
    if x_choice == "Use row index":
        x_values = np.arange(len(df), dtype=float)
        x_labels = [str(i) for i in range(len(df))]
        x_title = "row index"
    else:
        x_values, x_labels, x_title = parse_x_axis(df[x_choice])
except Exception as exc:
    st.error(str(exc))
    st.stop()

for target in targets:
    y_values = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    st.divider()
    st.header(f"분석 결과: {target}")

    with st.spinner("Classical Regression, Quantum HHL, CKKS + HHL 분석 중..."):
        try:
            result = run_analysis(x_values, y_values)
        except Exception as exc:
            st.error(f"분석 실패: {exc}")
            continue

    chart_df = build_chart_frame(x_values, y_values, x_labels, result)
    st.markdown('<div class="section-label">회귀 직선 비교</div>', unsafe_allow_html=True)
    render_chart(chart_df, x_title=x_title, y_title=target)

    st.markdown('<div class="section-label">방법별 수치 결과</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metrics("Classical Regression", result.get("classical", {}), include_stats=True)
    with col2:
        render_metrics("Quantum HHL", result.get("hhl", {}))
    with col3:
        render_metrics("CKKS + HHL", result.get("he_hhl", {}), include_stats=True)
        render_encrypted_preview(result.get("he_hhl", {}).get("encrypted_preview", {}))

    render_qhe_hhl_handoff(x_values, y_values)
