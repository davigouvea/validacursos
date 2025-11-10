import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ===================== Configuração básica =====================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "store")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

ALLOWED_EXT = {".csv"}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def dataset_path():
    """ Retorna o último csv salvo, se existir. """
    files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")])
    if not files:
        return None
    return os.path.join(UPLOAD_DIR, files[-1])

def load_df():
    path = dataset_path()
    if not path or not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df, path

# ===================== Gráficos (Plotly JSON) =====================
def fig_bar_counts(df, col):
    vc = df[col].value_counts().head(15)
    return {
        "data": [{
            "type": "bar",
            "x": vc.index.astype(str).tolist(),
            "y": vc.values.tolist(),
            "marker": {"color": "#2d9cdb"}
        }],
        "layout": {
            "title": f"Distribuição — {col}",
            "xaxis": {"title": col},
            "yaxis": {"title": "Contagem"},
            "margin": {"l":50,"r":20,"t":40,"b":80},
            "paper_bgcolor":"#0c121a", "plot_bgcolor":"#0c121a",
            "font":{"color":"#e9eef3"}
        }
    }

def fig_pie_share(df, col):
    vc = df[col].value_counts().head(8)
    return {
        "data": [{
            "type": "pie",
            "labels": vc.index.astype(str).tolist(),
            "values": vc.values.tolist(),
            "hole": 0.35
        }],
        "layout": {
            "title": f"Participação — {col}",
            "paper_bgcolor":"#0c121a",
            "font":{"color":"#e9eef3"}
        }
    }

def fig_scatter(df, x, y, color=None):
    data = []
    if color and color in df.columns:
        for g, d in df.groupby(color):
            data.append({
                "type": "scatter",
                "mode": "markers",
                "name": str(g),
                "x": d[x].tolist(),
                "y": d[y].tolist(),
                "marker": {"size": 9, "line": {"width": 0.5, "color": "#000"}}
            })
    else:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "name": "dados",
            "x": df[x].tolist(),
            "y": df[y].tolist(),
            "marker": {"size": 9, "color":"#27ae60", "line": {"width": 0.5, "color": "#000"}}
        })
    return {
        "data": data,
        "layout": {
            "title": f"Dispersão — {x} × {y}",
            "xaxis": {"title": x},
            "yaxis": {"title": y},
            "paper_bgcolor":"#0c121a","plot_bgcolor":"#0c121a",
            "font":{"color":"#e9eef3"}
        }
    }

def fig_corr_heatmap(df):
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return None
    corr = num.corr().round(2)
    return {
        "data": [{
            "type": "heatmap",
            "z": corr.values.tolist(),
            "x": corr.columns.tolist(),
            "y": corr.index.tolist(),
            "colorscale": "Viridis"
        }],
        "layout": {
            "title": "Correlação (numéricas)",
            "paper_bgcolor":"#0c121a","plot_bgcolor":"#0c121a",
            "font":{"color":"#e9eef3"}
        }
    }

# ===================== Treino / Predição ML =====================
def build_model(task, model_name):
    if task == "regressao":
        if model_name == "LinearRegression":
            return LinearRegression()
        if model_name == "DecisionTree":
            return DecisionTreeRegressor(random_state=42)
        if model_name == "KNN":
            return KNeighborsRegressor(n_neighbors=5)
        if model_name == "RandomForest":
            return RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        if model_name == "LogisticRegression":
            return LogisticRegression(max_iter=1000)
        if model_name == "DecisionTree":
            return DecisionTreeClassifier(random_state=42)
        if model_name == "KNN":
            return KNeighborsClassifier(n_neighbors=7)
        if model_name == "RandomForest":
            return RandomForestClassifier(n_estimators=300, random_state=42)
    raise ValueError("Modelo não suportado")

def infer_task(df, target_col):
    if pd.api.types.is_numeric_dtype(df[target_col]):
        return "regressao"
    return "classificacao"

def train_model(df, target_col, features, model_name):
    task = infer_task(df, target_col)
    X = df[features].copy()
    y = df[target_col].copy()

    # Identifica tipos
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
            ("cat", "passthrough", cat_cols)
        ],
        remainder="drop"
    )

    model = build_model(task, model_name)
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if task=="classificacao" else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if task == "regressao":
        metrics = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4)
        }
    else:
        if pd.api.types.is_numeric_dtype(y_test):
            y_bin = y_test
        else:
            y_bin = y_test
        metrics = {
            "acc": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4)
        }

    # Persistência
    model_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
    joblib.dump({
        "pipeline": pipe,
        "task": task,
        "target": target_col,
        "features": features,
        "metrics": metrics
    }, os.path.join(MODEL_DIR, f"{model_id}.joblib"))

    return model_id, metrics, task

def list_models():
    items = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".joblib"):
            p = os.path.join(MODEL_DIR, f)
            try:
                meta = joblib.load(p)
                items.append({
                    "id": f.replace(".joblib",""),
                    "target": meta.get("target"),
                    "features": meta.get("features"),
                    "task": meta.get("task"),
                    "metrics": meta.get("metrics")
                })
            except Exception:
                pass
    return sorted(items, key=lambda x: x["id"], reverse=True)

def predict_with_model(model_id, payload):
    pth = os.path.join(MODEL_DIR, f"{model_id}.joblib")
    if not os.path.exists(pth):
        return {"error": "modelo não encontrado"}
    meta = joblib.load(pth)
    pipe = meta["pipeline"]
    task = meta["task"]
    feats = meta["features"]

    X = pd.DataFrame([payload])[feats]
    pred = pipe.predict(X)[0]
    return {
        "task": task,
        "prediction": float(pred) if isinstance(pred, (np.floating, float, int)) else str(pred)
    }

# ===================== Flask =====================
app = Flask(__name__)
app.secret_key = "validacursos-secret"

@app.route("/", methods=["GET"])
def home():
    df, path = load_df()
    have_data = df is not None
    cols = df.columns.tolist() if have_data else []
    cat_cols = (df.select_dtypes(exclude=["number"]).columns.tolist() if have_data else [])
    num_cols = (df.select_dtypes(include=["number"]).columns.tolist() if have_data else [])
    return render_template("index.html",
                           have_data=have_data,
                           path=os.path.basename(path) if path else None,
                           cols=cols, cat_cols=cat_cols, num_cols=num_cols,
                           models=list_models())

@app.post("/upload")
def upload():
    if "file" not in request.files:
        flash("Nenhum arquivo enviado.", "danger")
        return redirect(url_for("home"))
    f = request.files["file"]
    if f.filename == "":
        flash("Selecione um arquivo .csv.", "warning")
        return redirect(url_for("home"))
    if not allowed_file(f.filename):
        flash("Apenas .csv é permitido.", "danger")
        return redirect(url_for("home"))
    dest = os.path.join(UPLOAD_DIR, f"dataset_{uuid.uuid4().hex[:8]}.csv")
    f.save(dest)
    flash("Upload realizado com sucesso. Dataset atualizado (modelo pode ser re-treinado).", "success")
    return redirect(url_for("home"))

@app.get("/dashboard")
def dashboard():
    df, _ = load_df()
    if df is None:
        flash("Carregue um CSV para visualizar o dashboard.", "warning")
        return redirect(url_for("home"))

    # Seleções padrão
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    # Gráficos default
    pie_json = json.dumps(fig_pie_share(df, cat_cols[0])) if cat_cols else None
    bar_json = json.dumps(fig_bar_counts(df, cat_cols[0])) if cat_cols else None
    scatter_json = json.dumps(
        fig_scatter(df, num_cols[0], num_cols[1], color=cat_cols[0] if cat_cols else None)
    ) if len(num_cols) >= 2 else None
    corr_json = json.dumps(fig_corr_heatmap(df)) if fig_corr_heatmap(df) else None

    return render_template("dashboard.html",
                           cols=all_cols, num_cols=num_cols, cat_cols=cat_cols,
                           pie_json=pie_json, bar_json=bar_json,
                           scatter_json=scatter_json, corr_json=corr_json,
                           models=list_models())

@app.post("/chart")
def chart():
    """Retorna JSON do gráfico solicitado (dinâmico p/ filtros)."""
    df, _ = load_df()
    if df is None: return jsonify({"error": "Sem dados"}), 400
    payload = request.get_json()
    kind = payload.get("kind")

    if kind == "bar":
        col = payload.get("col")
        return jsonify(fig_bar_counts(df, col))
    if kind == "pie":
        col = payload.get("col")
        return jsonify(fig_pie_share(df, col))
    if kind == "scatter":
        x = payload.get("x"); y = payload.get("y"); color = payload.get("color")
        return jsonify(fig_scatter(df, x, y, color))
    if kind == "corr":
        return jsonify(fig_corr_heatmap(df))
    return jsonify({"error":"gráfico não suportado"}), 400

@app.post("/train")
def train():
    df, _ = load_df()
    if df is None:
        return jsonify({"error": "Carregue um CSV primeiro."}), 400

    data = request.get_json()
    target = data.get("target")
    features = data.get("features", [])
    model_name = data.get("model", "RandomForest")

    missing = [c for c in [target, *features] if c not in df.columns]
    if missing:
        return jsonify({"error": f"Colunas inválidas: {missing}"}), 400

    dfc = df[[target, *features]].dropna().copy()
    model_id, metrics, task = train_model(dfc, target, features, model_name)
    return jsonify({"model_id": model_id, "metrics": metrics, "task": task})

@app.post("/predict")
def predict():
    data = request.get_json()
    model_id = data.get("model_id")
    payload  = data.get("payload", {})
    if not model_id:
        return jsonify({"error":"Informe model_id"}), 400
    return jsonify(predict_with_model(model_id, payload))

if __name__ == "__main__":
    # Rodar:  python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
