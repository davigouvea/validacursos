# ValidaCursos — Aplicação Web (Flask + ML)

Aplicação web para upload de CSV, análise visual (barras, pizza, dispersão, correlação) e Machine Learning (regressão ou classificação) com re-treinamento dinâmico.

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

## Como rodar
```bash
# 1) criar venv (opcional)
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) instalar deps
pip install -r requirements.txt

# 3) executar
python app.py
