# Deep-Learning-2026-MDSAA

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for Python environment management. The project is configured to automatically use Python 3.12.11 via the `.python-version` file.

### 1. Create the Environment

First, navigate to the project folder and create the virtual environment. `uv` will automatically detect the required Python version and download it if necessary:

```bash
uv venv
```

> **Note:** If you already see a `.venv` folder in your project directory, do not run `uv venv`. Skip directly to the activation step below.

### 2. Activate the Environment

Next, activate the environment based on your operating system:

* **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

Once the environment is activated (you should see `(.venv)` in your terminal prompt), install the required deep learning packages (TensorFlow, Keras, etc.):

```bash
uv pip install -r requirements.txt
```