#!/usr/bin/env python3
"""
hhl_col_regression.py

Column-wise spreadsheet analysis with Qiskit.

For each selected numeric target column, this script computes:
    * column mean
    * column variance
    * a simple linear regression model

Model per target column:
    y_i = intercept + slope * x_i

By default, x_i is the row index 0, 1, 2, ..., n-1.  For a statistically
more meaningful regression, pass a numeric predictor column with --x-column.
For example:
    score = intercept + slope * study_hours

The two regression coefficients are obtained by solving the 2x2 normal equation
with a compact HHL-style quantum circuit built with Qiskit.

Install:
    pip install qiskit qiskit-ibm-runtime pandas numpy scipy openpyxl

Examples:
    # Analyze every numeric column as y against the row index.
    python hhl_col_regression.py input.xlsx output_hhl_col.xlsx

    # Analyze selected target columns against a numeric predictor column.
    python hhl_col_regression.py input.xlsx output_hhl_col.xlsx --x-column study_hours --target-columns score,exam_score

    # Exclude identifier columns when numeric IDs should not be treated as targets.
    python hhl_col_regression.py input.xlsx output_hhl_col.xlsx --exclude-columns student_id,name

    # Optional one-time IBM Quantum credential save. Prefer an environment variable.
    # macOS/Linux: export QISKIT_IBM_TOKEN="your_api_key"
    # Windows PowerShell: $env:QISKIT_IBM_TOKEN="your_api_key"
    python hhl_col_regression.py --save-ibm-token --ibm-instance "your_instance_crn_or_name"

    # Run analysis and initialize saved IBM Runtime credentials.
    python hhl_col_regression.py input.xlsx output_hhl_col.xlsx --initialize-ibm-runtime

Notes:
    * HHL returns a normalized quantum state. This script reconstructs the two
      classical regression coefficients from the post-selected statevector.
    * Because this is simple linear regression, the normal equation has two
      unknowns: intercept and slope. Therefore the HHL circuit is a compact
      exact-eigenbasis implementation for a 2x2 system. It is useful for small
      educational and simulation workflows, not a practical quantum speedup claim.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "Missing dependency: qiskit. Install with: "
        "pip install qiskit qiskit-ibm-runtime pandas numpy scipy openpyxl"
    ) from exc

EPS = 1.0e-12

# Optional local placeholders for a private copy of this file.
# Safer alternatives are --ibm-token or the QISKIT_IBM_TOKEN environment variable.
DEFAULT_IBM_QUANTUM_TOKEN = "너의 API key"
DEFAULT_IBM_QUANTUM_INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/a9b97467e1764d528c81e4fb88381d3e:26765c43-dcdf-4298-8ee5-4cdcd07bfb68::"
DEFAULT_IBM_RUNTIME_CHANNEL = "ibm_quantum_platform"
DEFAULT_IBM_TOKEN_ENV = "QISKIT_IBM_TOKEN"
DEFAULT_IBM_INSTANCE_ENV = "QISKIT_IBM_INSTANCE"


@dataclass
class ColumnRegressionResult:
    target_column: str
    x_source: str
    n_points: int
    mean: float
    variance: float
    intercept: float
    slope: float
    r2: float
    solver_success: bool
    solver_message: str
    condition_number: float
    hhl_success_probability: float
    hhl_c: float


def parse_sheet_name(value: str) -> str | int:
    """Allow --sheet 0 for the first sheet, or --sheet SheetName."""
    try:
        return int(value)
    except ValueError:
        return value


def split_csv_text(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def parse_float_list(value: str | None) -> np.ndarray | None:
    if value is None or not value.strip():
        return None
    try:
        return np.array([float(item.strip()) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise ValueError("--x-values must be a comma-separated list of numbers.") from exc


def resolve_ibm_token(args: argparse.Namespace) -> str | None:
    """Resolve the IBM Quantum API token from CLI, env var, or private placeholder."""
    if getattr(args, "ibm_token", None):
        return str(args.ibm_token).strip()
    env_name = getattr(args, "ibm_token_env", DEFAULT_IBM_TOKEN_ENV)
    token = os.environ.get(env_name)
    if token:
        return token.strip()
    return DEFAULT_IBM_QUANTUM_TOKEN.strip() or None


def resolve_ibm_instance(args: argparse.Namespace) -> str | None:
    """Resolve the IBM Quantum instance from CLI, env var, or private placeholder."""
    if getattr(args, "ibm_instance", None):
        return str(args.ibm_instance).strip()
    env_name = getattr(args, "ibm_instance_env", DEFAULT_IBM_INSTANCE_ENV)
    instance = os.environ.get(env_name)
    if instance:
        return instance.strip()
    return DEFAULT_IBM_QUANTUM_INSTANCE.strip() or None


def configure_ibm_runtime_if_requested(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    """Optionally save or load IBM Quantum Runtime credentials.

    This column-wise HHL script runs its small quantum circuits through Qiskit
    local statevector simulation. IBM credentials are optional and are included
    so you can save/load your account in the same workflow.
    """
    wants_save = bool(getattr(args, "save_ibm_token", False))
    wants_init = bool(getattr(args, "initialize_ibm_runtime", False))
    if not (wants_save or wants_init):
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise SystemExit(
            "Missing dependency: qiskit-ibm-runtime. Install with: pip install qiskit-ibm-runtime"
        ) from exc

    token = resolve_ibm_token(args)
    instance = resolve_ibm_instance(args)

    if wants_save:
        if not token:
            parser.error(
                "--save-ibm-token requires --ibm-token, an environment variable, "
                "or DEFAULT_IBM_QUANTUM_TOKEN inside the file. "
                f"Default environment variable: {args.ibm_token_env}."
            )
        save_kwargs: dict[str, object] = {
            "token": token,
            "channel": args.ibm_channel,
            "set_as_default": True,
            "overwrite": bool(args.overwrite_ibm_account),
        }
        if instance:
            save_kwargs["instance"] = instance
        if args.ibm_account_name:
            save_kwargs["name"] = args.ibm_account_name
        QiskitRuntimeService.save_account(**save_kwargs)
        print("Saved IBM Quantum Runtime credentials for future Qiskit sessions.")

    if wants_init:
        init_kwargs: dict[str, object] = {"channel": args.ibm_channel}
        if token:
            init_kwargs["token"] = token
        elif args.ibm_account_name:
            init_kwargs["name"] = args.ibm_account_name
        if instance:
            init_kwargs["instance"] = instance
        service = QiskitRuntimeService(**init_kwargs)
        print("Initialized IBM Quantum Runtime service.")
        return service

    return None


def read_spreadsheet(path: Path, sheet: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm", ".ods"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported input file extension '{path.suffix}'. Use .xlsx, .xls, .xlsm, .ods, .csv, or .txt."
    )


def write_output(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        df.to_excel(path, index=False)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output path must end with .csv, .txt, .xlsx, or .xlsm.")


def make_x_values(
    df: pd.DataFrame,
    x_column: str | None,
    x_values_arg: str | None,
    x_mode: str,
) -> tuple[np.ndarray, str]:
    """Return predictor values and a readable description of their source."""
    if x_column is not None and x_values_arg is not None:
        raise ValueError("Use either --x-column or --x-values, not both.")

    if x_column is not None:
        if x_column not in df.columns:
            raise ValueError(f"--x-column '{x_column}' was not found in the input columns.")
        x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
        return x, f"column:{x_column}"

    explicit_x = parse_float_list(x_values_arg)
    if explicit_x is not None:
        if explicit_x.size != len(df):
            raise ValueError(
                f"--x-values has length {explicit_x.size}, but the spreadsheet has {len(df)} rows."
            )
        return explicit_x, "explicit:x-values"

    if x_mode == "row-number":
        return np.arange(1, len(df) + 1, dtype=float), "row-number"

    return np.arange(len(df), dtype=float), "row-index"


def infer_target_columns(
    df: pd.DataFrame,
    explicit_columns: Sequence[str] | None,
    exclude_columns: Sequence[str] | None,
    id_column: str | None,
    x_column: str | None,
) -> list[str]:
    """Choose target y-columns for column-wise regression."""
    if explicit_columns is not None:
        missing = [col for col in explicit_columns if col not in df.columns]
        if missing:
            raise ValueError(f"These --target-columns were not found: {missing}")
        return list(explicit_columns)

    excluded = set(exclude_columns or [])
    if id_column is not None:
        excluded.add(id_column)
    if x_column is not None:
        excluded.add(x_column)

    missing_excluded = [col for col in excluded if col not in df.columns]
    if missing_excluded:
        raise ValueError(f"These excluded columns were not found: {missing_excluded}")

    target_columns: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            target_columns.append(col)

    if not target_columns:
        raise ValueError(
            "Could not infer any numeric target columns. Use --target-columns to specify them explicitly."
        )
    return target_columns


def hhl_solve_2x2(
    matrix: np.ndarray,
    rhs: np.ndarray,
    c_factor: float = 0.9,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    """Solve a 2x2 Hermitian positive definite system with a compact HHL circuit.

    The circuit uses one system qubit and one post-selection ancilla:
        |b> -> eigenbasis -> controlled R_y eigenvalue inversion rotations
             -> original basis -> postselect ancilla=1.

    The post-selected branch is C * A^{-1}|b_normalized>. Multiplying by
    ||rhs|| / C recovers the classical 2-vector solution.
    """
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("hhl_solve_2x2 expects a 2x2 matrix.")
    if b.shape != (2,):
        raise ValueError("hhl_solve_2x2 expects a length-2 rhs vector.")
    if not 0.0 < c_factor <= 1.0:
        raise ValueError("c_factor must be in the interval (0, 1].")

    A = 0.5 * (A + A.T)
    rhs_norm = float(np.linalg.norm(b))
    if rhs_norm < EPS:
        return np.zeros(2), {
            "solver_success": True,
            "solver_message": "rhs is zero; solution is zero",
            "hhl_success_probability": 0.0,
            "hhl_c": 0.0,
        }

    eigvals, eigvecs = np.linalg.eigh(A)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    if min_eig <= EPS:
        raise ValueError(
            "HHL requires a positive definite matrix. Increase --ridge if the column is singular or ill-conditioned."
        )

    C = float(c_factor * min_eig)
    ratios = np.clip(C / eigvals, 0.0, 1.0)
    theta_0 = 2.0 * math.asin(float(ratios[0]))
    theta_1 = 2.0 * math.asin(float(ratios[1]))

    b_state = b / rhs_norm
    prep_angle = 2.0 * math.atan2(float(b_state[1]), float(b_state[0]))

    system = 0
    ancilla = 1
    qc = QuantumCircuit(2, name="compact_hhl_2x2")
    qc.ry(prep_angle, system)

    # np.linalg.eigh returns eigenvectors as columns. U maps eigenbasis
    # coordinates to the original computational basis. Therefore U.T moves
    # from original coordinates to the eigenbasis for real symmetric A.
    U = eigvecs.astype(complex)
    qc.unitary(U.conj().T, [system], label="to_eigenbasis")

    # Controlled eigenvalue inversion rotations.
    # Eigenvalue 0 is represented by system |0>; eigenvalue 1 by system |1>.
    qc.x(system)
    qc.cry(theta_0, system, ancilla)
    qc.x(system)
    qc.cry(theta_1, system, ancilla)

    qc.unitary(U, [system], label="from_eigenbasis")

    state = Statevector.from_instruction(qc)
    data = np.asarray(state.data, dtype=complex)

    # Qiskit stores two-qubit amplitudes in little-endian order. With q0 as
    # system and q1 as ancilla, ancilla=1 corresponds to indices |10>, |11>,
    # i.e. [2, 3], preserving system states |0>, |1>.
    postselected_branch = np.array([data[2], data[3]], dtype=complex)
    success_probability = float(np.real(np.vdot(postselected_branch, postselected_branch)))

    if success_probability < EPS:
        raise ValueError("HHL post-selection probability is numerically zero.")

    solution = (rhs_norm / C) * postselected_branch
    solution = np.real_if_close(solution, tol=1000).astype(float)

    return solution, {
        "solver_success": True,
        "solver_message": "ok",
        "hhl_success_probability": success_probability,
        "hhl_c": C,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
    }


def normal_equation_for_column(
    x_values: np.ndarray,
    y_values: np.ndarray,
    ridge: float,
    standardize_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return A, rhs, mask, x_center, x_scale for one target column."""
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[mask].astype(float)
    y = y_values[mask].astype(float)

    if x.size < 2:
        raise ValueError("fewer than two finite data points")
    if np.allclose(x, x[0]):
        raise ValueError("all x-values are identical")

    if standardize_x:
        x_center = float(np.mean(x))
        x_scale = float(np.std(x))
        if x_scale < EPS:
            raise ValueError("x standard deviation is zero")
        z = (x - x_center) / x_scale
    else:
        x_center = 0.0
        x_scale = 1.0
        z = x

    design = np.column_stack([np.ones_like(z), z])
    A = design.T @ design
    if ridge > 0.0:
        A = A + ridge * np.eye(2)
    rhs = design.T @ y
    return A, rhs, mask, x_center, x_scale


def convert_standardized_coefficients(
    beta_standardized: np.ndarray,
    x_center: float,
    x_scale: float,
) -> tuple[float, float]:
    beta_0_z, beta_1_z = float(beta_standardized[0]), float(beta_standardized[1])
    slope = beta_1_z / x_scale
    intercept = beta_0_z - beta_1_z * x_center / x_scale
    return intercept, slope


def r_squared(x: np.ndarray, y: np.ndarray, intercept: float, slope: float) -> float:
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def analyze_column(
    target_column: str,
    x_source: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    variance_ddof: int,
    ridge: float,
    standardize_x: bool,
    hhl_c_factor: float,
) -> ColumnRegressionResult:
    finite_y = y_values[np.isfinite(y_values)]
    mean = float(np.mean(finite_y)) if finite_y.size else float("nan")
    variance = (
        float(np.var(finite_y, ddof=variance_ddof))
        if finite_y.size > variance_ddof
        else float("nan")
    )

    try:
        A, rhs, mask, x_center, x_scale = normal_equation_for_column(
            x_values=x_values,
            y_values=y_values,
            ridge=ridge,
            standardize_x=standardize_x,
        )
        beta_z, meta = hhl_solve_2x2(A, rhs, c_factor=hhl_c_factor)
        intercept, slope = convert_standardized_coefficients(beta_z, x_center, x_scale)
        x_fit = x_values[mask].astype(float)
        y_fit = y_values[mask].astype(float)
        r2 = r_squared(x_fit, y_fit, intercept, slope)
        return ColumnRegressionResult(
            target_column=target_column,
            x_source=x_source,
            n_points=int(mask.sum()),
            mean=mean,
            variance=variance,
            intercept=intercept,
            slope=slope,
            r2=r2,
            solver_success=bool(meta["solver_success"]),
            solver_message=str(meta["solver_message"]),
            condition_number=float(np.linalg.cond(A)),
            hhl_success_probability=float(meta["hhl_success_probability"]),
            hhl_c=float(meta["hhl_c"]),
        )
    except Exception as exc:  # keep processing the remaining columns
        return ColumnRegressionResult(
            target_column=target_column,
            x_source=x_source,
            n_points=int(np.sum(np.isfinite(x_values) & np.isfinite(y_values))),
            mean=mean,
            variance=variance,
            intercept=float("nan"),
            slope=float("nan"),
            r2=float("nan"),
            solver_success=False,
            solver_message=str(exc),
            condition_number=float("nan"),
            hhl_success_probability=float("nan"),
            hhl_c=float("nan"),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Column-wise mean, variance, and simple linear regression using a compact Qiskit HHL solver."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input Excel/spreadsheet file: .xlsx, .xls, .xlsm, .ods, .csv, or .txt",
    )
    parser.add_argument("output", nargs="?", type=Path, help="Output file: .csv, .txt, .xlsx, or .xlsm")
    parser.add_argument("--sheet", default="0", help="Excel sheet index or name. Default: 0")
    parser.add_argument(
        "--target-columns",
        default=None,
        help="Comma-separated y columns to analyze. Default: infer all numeric columns except excluded/x columns.",
    )
    parser.add_argument(
        "--x-column",
        default=None,
        help="Numeric predictor column used as x. Default: row index 0,1,2,...",
    )
    parser.add_argument(
        "--x-values",
        default=None,
        help="Comma-separated x-values, one per spreadsheet row. Cannot be used with --x-column.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["index", "row-number"],
        default="index",
        help="How to create x-values when --x-column and --x-values are not supplied. Default: index.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Backward-compatible alias for an identifier column to exclude from automatic target inference.",
    )
    parser.add_argument(
        "--exclude-columns",
        default=None,
        help="Comma-separated columns to exclude from automatic target inference, such as IDs or names.",
    )
    parser.add_argument(
        "--variance-ddof",
        type=int,
        default=1,
        help="Variance delta degrees of freedom: 1 for sample variance, 0 for population variance. Default: 1",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0e-8,
        help="Small ridge term added to X^T X for positive definiteness. Default: 1e-8",
    )
    parser.add_argument(
        "--no-standardize-x",
        action="store_true",
        help="Disable internal x standardization before solving the normal equation.",
    )
    parser.add_argument(
        "--hhl-c-factor",
        type=float,
        default=0.9,
        help="Controlled-rotation constant C = factor * min_eigenvalue. Must be in (0,1]. Default: 0.9",
    )

    ibm_auth = parser.add_argument_group(
        "IBM Quantum / Qiskit Runtime credential options",
        "Optional. The column-wise solver uses local Qiskit Statevector simulation, "
        "but these options let you save or initialize your IBM Quantum account.",
    )
    ibm_auth.add_argument(
        "--ibm-token",
        default=None,
        help="IBM Quantum API token. Safer alternative: set QISKIT_IBM_TOKEN and use --save-ibm-token.",
    )
    ibm_auth.add_argument(
        "--ibm-token-env",
        default=DEFAULT_IBM_TOKEN_ENV,
        help=f"Environment variable that stores the IBM Quantum API token. Default: {DEFAULT_IBM_TOKEN_ENV}",
    )
    ibm_auth.add_argument(
        "--ibm-instance",
        default=None,
        help="Optional IBM Quantum instance CRN or instance name.",
    )
    ibm_auth.add_argument(
        "--ibm-instance-env",
        default=DEFAULT_IBM_INSTANCE_ENV,
        help=f"Environment variable that stores the IBM Quantum instance. Default: {DEFAULT_IBM_INSTANCE_ENV}",
    )
    ibm_auth.add_argument(
        "--ibm-account-name",
        default=None,
        help="Optional local saved-account name for QiskitRuntimeService.",
    )
    ibm_auth.add_argument(
        "--ibm-channel",
        default=DEFAULT_IBM_RUNTIME_CHANNEL,
        choices=["ibm_quantum_platform", "ibm_cloud"],
        help=f"IBM Runtime channel. Default: {DEFAULT_IBM_RUNTIME_CHANNEL}",
    )
    ibm_auth.add_argument(
        "--save-ibm-token",
        action="store_true",
        help="Save the IBM Quantum API token locally using QiskitRuntimeService.save_account().",
    )
    ibm_auth.add_argument(
        "--overwrite-ibm-account",
        action="store_true",
        help="Overwrite an existing saved IBM Quantum account with the same name/default slot.",
    )
    ibm_auth.add_argument(
        "--initialize-ibm-runtime",
        action="store_true",
        help="Initialize QiskitRuntimeService before processing. This is optional for local simulation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.variance_ddof < 0:
        parser.error("--variance-ddof must be non-negative.")
    if args.ridge < 0.0:
        parser.error("--ridge must be non-negative.")
    if not 0.0 < args.hhl_c_factor <= 1.0:
        parser.error("--hhl-c-factor must be in the interval (0, 1].")

    configure_ibm_runtime_if_requested(args, parser)

    if args.input is None or args.output is None:
        if args.save_ibm_token or args.initialize_ibm_runtime:
            return 0
        parser.error("input and output files are required unless you only save/initialize IBM credentials.")

    df = read_spreadsheet(args.input, parse_sheet_name(args.sheet))

    if args.id_column is not None and args.id_column not in df.columns:
        parser.error(f"--id-column '{args.id_column}' was not found in the input columns.")

    try:
        x_values, x_source = make_x_values(
            df=df,
            x_column=args.x_column,
            x_values_arg=args.x_values,
            x_mode=args.x_mode,
        )
        target_columns = infer_target_columns(
            df=df,
            explicit_columns=split_csv_text(args.target_columns),
            exclude_columns=split_csv_text(args.exclude_columns),
            id_column=args.id_column,
            x_column=args.x_column,
        )
    except ValueError as exc:
        parser.error(str(exc))

    numeric_targets = df[target_columns].apply(pd.to_numeric, errors="coerce")

    records: list[dict[str, object]] = []
    for target_column in target_columns:
        y_values = numeric_targets[target_column].to_numpy(dtype=float)
        result = analyze_column(
            target_column=target_column,
            x_source=x_source,
            x_values=x_values,
            y_values=y_values,
            variance_ddof=args.variance_ddof,
            ridge=args.ridge,
            standardize_x=not args.no_standardize_x,
            hhl_c_factor=args.hhl_c_factor,
        )
        records.append(result.__dict__.copy())

    out = pd.DataFrame.from_records(records)
    write_output(out, args.output)
    print(f"Wrote {len(out)} column results to {args.output}")
    print(f"Used x source: {x_source}")
    print(f"Used target columns: {', '.join(map(str, target_columns))}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
