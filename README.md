# Quantum Homomorphic Encryption (HE) Regression Web App

This project is a web application that performs statistical analysis (mean, variance, and simple linear regression) on spreadsheet data. It compares results from three different computational approaches:

1. **Classical Computation**: Standard linear algebra calculations.
2. **Quantum Algorithm (HHL)**: Uses Qiskit to solve the normal equation via the HHL (Harrow-Hassidim-Lloyd) quantum algorithm for linear systems.
3. **Homomorphic Encryption (TenSEAL CKKS) + HHL**: Encrypts the dataset using the CKKS homomorphic encryption scheme, homomorphically computes the components of the normal equation ($X^T X$ and $X^T y$), decrypts them, and then feeds them into the HHL quantum circuit to find the regression coefficients.

## Folder Structure

- **`backend/`**: Contains the FastAPI Python backend (`main.py`) which orchestrates the analysis and serves the frontend. It also houses `he_hhl_solver.py`, our custom logic bridging TenSEAL CKKS encryption with Qiskit's HHL solver.
- **`frontend/`**: Contains a self-contained, beautifully designed Vanilla HTML/JS/CSS web interface. The frontend logic dynamically communicates with the backend without requiring Node.js or `npm`.
- **`data/`**: Contains the sample dataset (`example_quantum_chip_daily.csv`) and output examples.
- **`docs/`**: Contains the original project requirements and the development plan.
- **`scripts/`**: Contains the original, standalone Qiskit regression scripts (`hhl_col_regression.py` and `vqls_col_regression.py`) provided as the foundation for the quantum solver.

## Setup Instructions

To run this app locally, you only need Python installed on your machine. The frontend is served directly by the Python backend.

### 1. Create a Virtual Environment (Recommended)

Open your terminal and navigate to the `backend/` directory:
```bash
cd backend
python -m venv venv
```

Activate the virtual environment:
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

### 2. Install Dependencies

Install the required Python packages (including FastAPI, Uvicorn, TenSEAL, and Qiskit):
```bash
pip install -r requirements.txt
```

### 3. Run the Web Server

Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the App

Open your favorite web browser and navigate to:
**http://localhost:8000**

From here, you can upload your CSV/Excel files (like the one in the `data/` folder), optionally specify predictor and target columns, and click **Analyze Data** to run the regression models through the classical, quantum, and homomorphically encrypted layers!
