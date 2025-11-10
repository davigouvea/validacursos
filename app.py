import os
import json
import uuid
import joblib
import sqlite3
import numpy as np
import pandas as pd
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, session
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ===================== Config básica e caminhos =====================
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "store")
DB_PATH    = os.path.join(BASE_DIR, "data", "app.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

ALLOWED_EXT = {".csv"}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def dataset_path():
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

# ===================== DB de usuários (SQLite) =====================
def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        );
        """)
        # seed admin
        cur.execute("SELECT id FROM users WHERE email=?", ("admin@local",))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                ("Administrador", "admin@local", generate_password_hash("admin123"), "admin")
            )
        con.commit()

def find_user_by_email(email):
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("SELECT id, name, email, password_hash, role FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "email": row[2], "password_hash": row[3], "role": row[4]}
        return None

def find_user_by_id(uid):
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("SELECT id, name, email, role FROM users WHERE id=?", (uid,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "email": row[2], "role": row[3]}
        return None

# ===================== Auth helpers =====================
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Faça login para acessar.", "warning")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapper

def current_user():
    uid = session.get("user_id")
    return find_user_by_id(uid) if uid else None

# ===================== Plotly (JSON) =====================
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
            "xaxis": {"title": col}, "yaxis": {"title": "Contagem"},
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
            "paper_bgcolor":"#0c121a", "font":{"color":"#e9eef3"}
        }
    }

def fig_scatter(df, x, y, color=None):
    data = []
    if color and color in df.columns:
        for g, d in df.groupby(color):
            data.append({
                "type": "scatter", "mode": "markers", "name": str(g),
                "x": d[x].tolist(), "y": d[y].tolist(),
                "marker": {"size": 9, "line": {"width": 0.5, "color": "#000"}}
            })
    else:
        data.append({
            "type": "scatter", "mode": "markers", "name": "dados",
            "x": df[x].tolist(), "y": df[y].tolist(),
            "marker": {"size": 9, "color":"#27ae60", "line": {"width": 0.5, "color": "#000"}}
        })
    return {
        "data": data,
        "layout": {
            "title": f"Dispersão — {x} × {y}",
            "xaxis": {"title": x}, "yaxis": {"title": y},
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

# ===================== ML =====================
def build_model(task, model_name):
    if task == "regressao":
        if model_name == "LinearRegression": return LinearRegression()
        if model_name == "DecisionTree":     return DecisionTreeRegressor(random_state=42)
        if model_name == "KNN":              return KNeighborsRegressor(n_neighbors=5)
        if model_name == "RandomForest":     return RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        if model_name == "LogisticRegression": return LogisticRegression(max_iter=1000)
        if model_name == "DecisionTree":       return DecisionTreeClassifier(random_state=42)
        if model_name == "KNN":                return KNeighborsClassifier(n_neighbors=7)
        if model_name == "RandomForest":       return RandomForestClassifier(n_estimators=300, random_state=42)
    raise ValueError("Modelo não suportado")

def infer_task(df, target_col):
    return "regressao" if pd.api.types.is_numeric_dtype(df[target_col]) else "classificacao"

def train_model(df, target_col, features, model_name):
    task = infer_task(df, target_col)
    X = df[features].copy()
    y = df[target_col].copy()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if task=="classificacao" else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if task == "regressao":
        metrics = {"r2": round(float(r2_score(y_test, y_pred)),4),
                   "mae": round(float(mean_absolute_error(y_test, y_pred)),4)}
    else:
        metrics = {"acc": round(float(accuracy_score(y_test, y_pred)),4),
                   "f1":  round(float(f1_score(y_test, y_pred, average="weighted")),4)}

    model_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
    joblib.dump({
        "pipeline": pipe, "task": task, "target": target_col,
        "features": features, "metrics": metrics
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
    pipe = meta["pipeline"]; feats = meta["features"]; task = meta["task"]
    X = pd.DataFrame([payload])[feats]
    pred = pipe.predict(X)[0]
    return {"task": task, "prediction": float(pred) if isinstance(pred, (np.floating, float, int)) else str(pred)}

# ===================== Flask App =====================
app = Flask(__name__)
app.secret_key = "validacursos-secret"
init_db()  # cria DB e usuário admin

# -------- Autenticação --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        user = find_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], pwd):
            flash("Credenciais inválidas.", "danger")
            return redirect(url_for("login"))
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_role"] = user["role"]
        flash(f"Bem-vindo, {user['name']}!", "success")
        return redirect(request.args.get("next") or url_for("home"))
    return render_template("auth_login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        if not name or not email or not pwd:
            flash("Preencha todos os campos.", "warning")
            return redirect(url_for("register"))
        if find_user_by_email(email):
            flash("E-mail já cadastrado.", "danger")
            return redirect(url_for("register"))
        with get_conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name, email, generate_password_hash(pwd), "user")
            )
            con.commit()
        flash("Cadastro realizado. Faça login.", "success")
        return redirect(url_for("login"))
    return render_template("auth_register.html")

@app.get("/logout")
def logout():
    session.clear()
    flash("Sessão encerrada.", "info")
    return redirect(url_for("login"))

# -------- Páginas principais (protegidas) --------
@app.route("/", methods=["GET"])
@login_required
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
                           models=list_models(), user=current_user())

@app.post("/upload")
@login_required
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
    flash("Upload realizado com sucesso. Dataset atualizado.", "success")
    return redirect(url_for("home"))

@app.get("/dashboard")
@login_required
def dashboard():
    df, _ = load_df()
    if df is None:
        flash("Carregue um CSV para visualizar o dashboard.", "warning")
        return redirect(url_for("home"))

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    pie_json = json.dumps(fig_pie_share(df, cat_cols[0])) if cat_cols else None
    bar_json = json.dumps(fig_bar_counts(df, cat_cols[0])) if cat_cols else None
    scatter_json = json.dumps(fig_scatter(df, num_cols[0], num_cols[1], color=cat_cols[0] if cat_cols else None)) if len(num_cols)>=2 else None
    corr = fig_corr_heatmap(df)
    corr_json = json.dumps(corr) if corr else None

    return render_template("dashboard.html",
                           cols=all_cols, num_cols=num_cols, cat_cols=cat_cols,
                           pie_json=pie_json, bar_json=bar_json,
                           scatter_json=scatter_json, corr_json=corr_json,
                           models=list_models(), user=current_user())

@app.post("/chart")
@login_required
def chart():
    df, _ = load_df()
    if df is None: return jsonify({"error": "Sem dados"}), 400
    payload = request.get_json()
    kind = payload.get("kind")
    if kind == "bar":     return jsonify(fig_bar_counts(df, payload.get("col")))
    if kind == "pie":     return jsonify(fig_pie_share(df,  payload.get("col")))
    if kind == "scatter": return jsonify(fig_scatter(df, payload.get("x"), payload.get("y"), payload.get("color")))
    if kind == "corr":    return jsonify(fig_corr_heatmap(df))
    return jsonify({"error":"gráfico não suportado"}), 400

@app.post("/train")
@login_required
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
@login_required
def predict():
    data = request.get_json()
    model_id = data.get("model_id")
    payload  = data.get("payload", {})
    if not model_id:
        return jsonify({"error":"Informe model_id"}), 400
    return jsonify(predict_with_model(model_id, payload))

# ---- 404/500 simples ----
@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", **{"message":"Página não encontrada."}), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("base.html", **{"message":"Erro interno. Tente novamente."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
