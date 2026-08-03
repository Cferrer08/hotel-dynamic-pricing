import importlib

packages = [
    "pandas", "numpy", "faker", "sqlalchemy", "psycopg2",
    "sklearn", "statsmodels", "xgboost", "catboost", "shap",
    "matplotlib", "plotly", "streamlit", "fastapi",
]

for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "ok")
        print(f"{name:<14} {version}")
    except ImportError as e:
        print(f"{name:<14} MISSING -> {e}")