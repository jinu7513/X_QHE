import math
import base64
import hashlib
import numpy as np
import tenseal as ts
import sys
import os

# Add scripts directory to sys.path to access the teammate's script
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from hhl_col_regression import hhl_solve_2x2, convert_standardized_coefficients

def setup_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def ckks_ciphertext_preview(name: str, encrypted_vector, value_count: int, max_chars: int = 120) -> dict:
    serialized = encrypted_vector.serialize()
    encoded = base64.b64encode(serialized).decode("ascii")
    return {
        "name": name,
        "scheme": "CKKS",
        "value_count": int(value_count),
        "ciphertext_bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "base64_preview": encoded[:max_chars],
        "truncated": len(encoded) > max_chars,
    }

def r_squared(x: np.ndarray, y: np.ndarray, intercept: float, slope: float) -> float:
    if math.isnan(intercept) or math.isnan(slope):
        return float("nan")
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def run_analysis(x_values: np.ndarray, y_values: np.ndarray, ridge: float = 1.0e-8, hhl_c_factor: float = 0.9):
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[mask].astype(float)
    y = y_values[mask].astype(float)
    n = len(x)
    
    if n < 2:
        raise ValueError("fewer than two finite data points")
    if np.allclose(x, x[0]):
        raise ValueError("all x-values are identical")

    results = {}
    
    # 1. Classical Method
    x_mean_class = np.mean(x)
    y_mean_class = np.mean(y)
    y_var_class = np.var(y, ddof=1)
    
    x_center = float(np.mean(x))
    x_scale = float(np.std(x))
    z = (x - x_center) / x_scale
    
    design_z = np.column_stack([np.ones_like(z), z])
    A_z = design_z.T @ design_z + ridge * np.eye(2)
    rhs_z = design_z.T @ y
    
    beta_z_class = np.linalg.solve(A_z, rhs_z)
    intercept_class, slope_class = convert_standardized_coefficients(beta_z_class, x_center, x_scale)
    r2_class = r_squared(x, y, intercept_class, slope_class)
    
    results['classical'] = {
        'mean_y': float(y_mean_class),
        'var_y': float(y_var_class),
        'intercept': float(intercept_class),
        'slope': float(slope_class),
        'r2': float(r2_class)
    }
    
    # 2. HHL Method
    try:
        beta_hhl_z, _ = hhl_solve_2x2(A_z, rhs_z, c_factor=hhl_c_factor)
        intercept_hhl, slope_hhl = convert_standardized_coefficients(beta_hhl_z, x_center, x_scale)
        r2_hhl = r_squared(x, y, intercept_hhl, slope_hhl)
    except Exception as e:
        print(f"HHL Error: {e}")
        intercept_hhl, slope_hhl, r2_hhl = float('nan'), float('nan'), float('nan')

    results['hhl'] = {
        'intercept': float(intercept_hhl),
        'slope': float(slope_hhl),
        'r2': float(r2_hhl)
    }
    
    # 3. HE + HHL Method
    try:
        ctx = setup_tenseal_context()
        ones = np.ones_like(z).tolist()
        
        enc_z = ts.ckks_vector(ctx, z.tolist())
        enc_y = ts.ckks_vector(ctx, y.tolist())
        encrypted_preview = {
            "vectors": [
                ckks_ciphertext_preview("standardized x values", enc_z, n),
                ckks_ciphertext_preview("target y values", enc_y, n),
            ],
            "aggregates": []
        }
        
        # Homomorphic aggregations
        enc_sum_y = enc_y.dot(ones)
        enc_sum_y_sq = (enc_y * enc_y).dot(ones)
        
        enc_sum_z = enc_z.dot(ones)
        enc_sum_z_sq = (enc_z * enc_z).dot(ones)
        enc_sum_zy = (enc_z * enc_y).dot(ones)
        encrypted_preview["aggregates"] = [
            ckks_ciphertext_preview("sum(y)", enc_sum_y, 1),
            ckks_ciphertext_preview("sum(y^2)", enc_sum_y_sq, 1),
            ckks_ciphertext_preview("sum(z)", enc_sum_z, 1),
            ckks_ciphertext_preview("sum(z^2)", enc_sum_z_sq, 1),
            ckks_ciphertext_preview("sum(z*y)", enc_sum_zy, 1),
        ]
        
        # Decrypt
        dec_sum_y = enc_sum_y.decrypt()[0]
        dec_sum_y_sq = enc_sum_y_sq.decrypt()[0]
        
        dec_sum_z = enc_sum_z.decrypt()[0]
        dec_sum_z_sq = enc_sum_z_sq.decrypt()[0]
        dec_sum_zy = enc_sum_zy.decrypt()[0]
        
        mean_y_he = dec_sum_y / n
        var_y_he = (dec_sum_y_sq - (dec_sum_y**2)/n) / (n - 1) if n > 1 else 0.0
        
        A_he = np.array([
            [n, dec_sum_z],
            [dec_sum_z, dec_sum_z_sq]
        ]) + ridge * np.eye(2)
        rhs_he = np.array([dec_sum_y, dec_sum_zy])
        
        beta_he_hhl_z, _ = hhl_solve_2x2(A_he, rhs_he, c_factor=hhl_c_factor)
        intercept_he_hhl, slope_he_hhl = convert_standardized_coefficients(beta_he_hhl_z, x_center, x_scale)
        r2_he_hhl = r_squared(x, y, intercept_he_hhl, slope_he_hhl)
        
    except Exception as e:
        print(f"HE+HHL Error: {e}")
        mean_y_he, var_y_he = float('nan'), float('nan')
        intercept_he_hhl, slope_he_hhl, r2_he_hhl = float('nan'), float('nan'), float('nan')
        encrypted_preview = {"vectors": [], "aggregates": []}

    results['he_hhl'] = {
        'mean_y': float(mean_y_he),
        'var_y': float(var_y_he),
        'intercept': float(intercept_he_hhl),
        'slope': float(slope_he_hhl),
        'r2': float(r2_he_hhl),
        'encrypted_preview': encrypted_preview
    }

    return results
