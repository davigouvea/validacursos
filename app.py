import os
import json
import uuid
import sqlite3
import joblib
import numpy as np
import pandas as pd
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, session, Response
)

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ===================== CONFIG GERAL =====================
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "store")
DB_PATH    = os.path.join(DATA_DIR, "app.db")

for p in (DATA_DIR, UPLOAD_DIR, MODEL_DIR):
    os.makedirs(p, exist_ok=True)

ALLOWED_EXT = {".csv"}

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def dataset_path():
    files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")])
    return os.path.join(UPLOAD_DIR, files[-1]) if files else None

def load_df():
    path = dataset_path()
    if not path or not os.path.exists(path):
        return None, None
    return pd.read_csv(path), path


# ===================== BANCO (SQLite) =====================
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS consultations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          model_id TEXT NOT NULL,
          target TEXT NOT NULL,
          task TEXT NOT NULL,
          input_json TEXT NOT NULL,
          output TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        # admin seed
        cur.execute("SELECT id FROM users WHERE email='admin@local'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (name,email,password_hash,role) VALUES (?,?,?,?)",
                ("Administrador", "admin@local", generate_password_hash("admin123"), "admin")
            )
        con.commit()

def find_user_by_email(email):
    with get_conn() as con:
        row = con.execute("SELECT id,name,email,password_hash,role FROM users WHERE email=?", (email,)).fetchone()
        if row:
            return {"id":row[0],"name":row[1],"email":row[2],"password_hash":row[3],"role":row[4]}

def find_user_by_id(uid):
    if not uid: return None
    with get_conn() as con:
        row = con.execute("SELECT id,name,email,role FROM users WHERE id=?", (uid,)).fetchone()
        if row:
            return {"id":row[0],"name":row[1],"email":row[2],"role":row[3]}

def log_consultation(user_id, model_id, target, task, input_dict, output_val):
    with get_conn() as con:
        con.execute("""
        INSERT INTO consultations (user_id,model_id,target,task,input_json,output)
        VALUES (?,?,?,?,?,?)
        """, (user_id, model_id, target, task, json.dumps(input_dict, ensure_ascii=False), str(output_val)))
        con.commit()

def list_consultations(limit=500):
    with get_conn() as con:
        rows = con.execute("""
        SELECT c.id,c.created_at,u.name,u.email,c.model_id,c.target,c.task,c.input_json,c.output
        FROM consultations c JOIN users u ON u.id=c.user_id
        ORDER BY c.id DESC LIMIT ?
        """, (limit,)).fetchall()
        cols = ["id","created_at","user_name","user_email","model_id","target","task","input_json","output"]
        return [dict(zip(cols, r)) for r in rows]


# ===================== AUTH HELPERS =====================
def login_required(view):
    @wraps(view)
    def w(*a, **k):
        if not session.get("user_id"):
            flash("Faça login para continuar.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*a, **k)
    return w

def admin_required(view):
    @wraps(view)
    def w(*a, **k):
        if not session.get("user_id"):
            flash("Faça login para continuar.", "warning")
            return redirect(url_for("login", next=request.path))
        if session.get("user_role") != "admin":
            flash("Acesso restrito ao administrador.", "danger")
            return redirect(url_for("analisar"))
        return view(*a, **k)
    return w

def current_user():
    return find_user_by_id(session.get("user_id"))

def is_admin():
    return session.get("user_role") == "admin"


# ===================== PLOTS (Plotly JSON) =====================
def fig_bar_counts(df, col):
    vc = df[col].value_counts().head(15)
    return {
        "data": [{
            "type": "bar",
            "x": vc.index.astype(str).tolist(),
            "y": vc.values.tolist()
        }],
        "layout": {"title": f"Distribuição — {col}"}
    }

def fig_pie_share(df, col):
    vc = df[col].value_counts().head(8)
    return {
        "data": [{
            "type":"pie", "labels":vc.index.astype(str).tolist(), "values":vc.values.tolist(), "hole":0.35
        }],
        "layout": {"title": f"Participação — {col}"}
    }

def fig_scatter(df, x, y, color=None):
    data=[]
    if color and color in df.columns:
        for g, d in df.groupby(color):
            data.append({"type":"scatter","mode":"markers","name":str(g),"x":d[x].tolist(),"y":d[y].tolist()})
    else:
        data.append({"type":"scatter","mode":"markers","name":"dados","x":df[x].tolist(),"y":df[y].tolist()})
    return {"data":data,"layout":{"title":f"Dispersão — {x} × {y}","xaxis":{"title":x},"yaxis":{"title":y}}}

def fig_corr_heatmap(df):
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2: return None
    corr = num.corr().round(2)
    return {
        "data": [{
            "type":"heatmap","z":corr.values.tolist(),"x":corr.columns.tolist(),"y":corr.index.tolist(),"colorscale":"Viridis"
        }],
        "layout":{"title":"Correlação (numéricas)"}
    }


# ===================== ML =====================
def build_model(task, name):
    if task == "regressao":
        if name == "LinearRegression": return LinearRegression()
        if name == "DecisionTree":     return DecisionTreeRegressor(random_state=42)
        if name == "KNN":              return KNeighborsRegressor(n_neighbors=5)
        if name == "RandomForest":     return RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        if name == "LogisticRegression": return LogisticRegression(max_iter=1000)
        if name == "DecisionTree":       return DecisionTreeClassifier(random_state=42)
        if name == "KNN":                return KNeighborsClassifier(n_neighbors=7)
        if name == "RandomForest":       return RandomForestClassifier(n_estimators=300, random_state=42)
    raise ValueError("Modelo não suportado")

def infer_task(df, target):
    return "regressao" if pd.api.types.is_numeric_dtype(df[target]) else "classificacao"

def train_model(df, target, features, model_name):
    # garante tipos consistentes
    task = infer_task(df, target)
    X = df[features].copy()
    y = df[target].copy()

    # classificação: manter rótulos como string evita problemas com números/categorias mistos
    if task == "classificacao":
        y = y.astype(str)

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # transforma categóricas em string explícita (evita NaN/None inconsistentes)
    for c in cat_cols:
        X[c] = X[c].astype(str)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ],
        remainder="drop"
    )

    model = build_model(task, model_name)
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=(y if task=="classificacao" else None)
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
        "pipeline": pipe, "task": task, "target": target,
        "features": features, "metrics": metrics
    }, os.path.join(MODEL_DIR, f"{model_id}.joblib"))
    return model_id, metrics, task

def list_models():
    items=[]
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        return items
    for f in os.listdir(MODEL_DIR):
        if not f.endswith(".joblib"): continue
        p = os.path.join(MODEL_DIR, f)
        try:
            meta = joblib.load(p)
            items.append({
                "id": f[:-7],
                "target": meta.get("target"),
                "features": meta.get("features"),
                "task": meta.get("task"),
                "metrics": meta.get("metrics", {})
            })
        except Exception:
            continue
    items.sort(key=lambda m: os.path.getmtime(os.path.join(MODEL_DIR, m["id"] + ".joblib")), reverse=True)
    return items


# ===================== FLASK APP =====================
app = Flask(__name__)
app.secret_key = "validacursos-secret"
init_db()

# -------- AUTH --------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        user = find_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], pwd):
            flash("Credenciais inválidas.", "danger")
            return redirect(url_for("login"))
        session["user_id"], session["user_name"], session["user_role"] = user["id"], user["name"], user["role"]
        return redirect(request.args.get("next") or url_for("home"))
    return render_template("auth_login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        if not name or not email or not pwd:
            flash("Preencha todos os campos.", "warning"); return redirect(url_for("register"))
        if find_user_by_email(email):
            flash("E-mail já cadastrado.", "danger"); return redirect(url_for("register"))
        with get_conn() as con:
            con.execute("INSERT INTO users (name,email,password_hash,role) VALUES (?,?,?,?)",
                        (name,email,generate_password_hash(pwd),"user"))
            con.commit()
        flash("Cadastro realizado. Faça login.", "success")
        return redirect(url_for("login"))
    return render_template("auth_register.html")

@app.get("/logout")
def logout():
    session.clear()
    flash("Sessão encerrada.", "info")
    return redirect(url_for("login"))


# -------- HOME --------
@app.route("/")
@login_required
def home():
    df, path = load_df()
    have_data = df is not None
    cols = df.columns.tolist() if have_data else []
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist() if have_data else []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist() if have_data else []
    return render_template("index.html",
        have_data=have_data,
        path=os.path.basename(path) if path else None,
        cols=cols, cat_cols=cat_cols, num_cols=num_cols,
        models=list_models(), user=current_user(), is_admin=is_admin()
    )


# -------- UPLOAD (admin) --------
@app.post("/upload")
@admin_required
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Selecione um arquivo CSV.", "warning"); return redirect(url_for("home"))
    if not allowed_file(f.filename):
        flash("Apenas .csv é permitido.", "danger"); return redirect(url_for("home"))
    dest = os.path.join(UPLOAD_DIR, f"dataset_{uuid.uuid4().hex[:8]}.csv")
    f.save(dest)
    flash("Upload realizado com sucesso!", "success")
    return redirect(url_for("home"))


# -------- DASHBOARD (admin) --------
@app.get("/dashboard")
@admin_required
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
    scatter_json = (
        json.dumps(fig_scatter(df, num_cols[0], num_cols[1], color=cat_cols[0] if cat_cols else None))
        if len(num_cols) >= 2 else None
    )
    corr = fig_corr_heatmap(df)
    corr_json = json.dumps(corr) if corr else None

    return render_template(
        "dashboard.html",
        cols=all_cols, num_cols=num_cols, cat_cols=cat_cols,
        pie_json=pie_json, bar_json=bar_json,
        scatter_json=scatter_json, corr_json=corr_json,
        models=list_models(), user=current_user(), is_admin=True
    )


# -------- CHART API (admin) --------
@app.post("/chart")
@admin_required
def chart():
    df, _ = load_df()
    if df is None: return jsonify({"error":"Sem dados"}), 400
    p = request.get_json()
    kind = p.get("kind")
    if kind == "bar":     return jsonify(fig_bar_counts(df, p.get("col")))
    if kind == "pie":     return jsonify(fig_pie_share(df,  p.get("col")))
    if kind == "scatter": return jsonify(fig_scatter(df, p.get("x"), p.get("y"), p.get("color")))
    if kind == "corr":
        res = fig_corr_heatmap(df)
        return jsonify(res if res else {"data": [], "layout": {"title":"Correlação"}})
    return jsonify({"error":"gráfico não suportado"}), 400


# -------- TRAIN API (admin) --------
@app.post("/train")
@admin_required
def train():
    df, _ = load_df()
    if df is None: return jsonify({"error":"Carregue um CSV primeiro."}), 400
    data = request.get_json()
    target   = data.get("target")
    features = data.get("features", [])
    model    = data.get("model", "RandomForest")

    if not target or not features:
        return jsonify({"error": "Informe alvo e pelo menos uma feature."}), 400

    missing = [c for c in [target, *features] if c not in df.columns]
    if missing: return jsonify({"error": f"Colunas inválidas: {missing}"}), 400

    dfc = df[[target, *features]].copy()
    # remove linhas totalmente vazias nas colunas relevantes
    dfc = dfc.dropna(how="all")

    try:
        model_id, metrics, task = train_model(dfc, target, features, model)
        return jsonify({"model_id": model_id, "metrics": metrics, "task": task})
    except Exception as e:
        return jsonify({"error": f"Falha ao treinar: {e}"}), 500


# -------- ANALISAR (usuário) --------
@app.route("/analisar", methods=["GET","POST"])
@login_required
def analisar():
    df, _ = load_df()
    modelos = list_models()
    if df is None or not modelos:
        flash("O sistema ainda não foi preparado pelo administrador.", "warning")
        return redirect(url_for("home"))

    model_id = modelos[0]["id"]
    alvo     = modelos[0]["target"]
    feats    = modelos[0]["features"]

    meta = joblib.load(os.path.join(MODEL_DIR, f"{model_id}.joblib"))
    pipeline = meta["pipeline"]

    # sugestões
    sugestoes={}
    for f in feats:
        if f not in df.columns: continue
        col=df[f]
        if pd.api.types.is_numeric_dtype(col):
            sugestoes[f] = {"tipo":"num","min":float(col.min()),"max":float(col.max()),"media":float(col.mean())}
        else:
            sugestoes[f] = {"tipo":"cat","opcoes":sorted(map(str,col.dropna().unique().tolist()))[:30]}

    resultado=None
    if request.method=="POST":
        entrada={}
        for f in feats:
            v = request.form.get(f,"")
            if v=="":
                v=None
            else:
                # se a sugestão for numérica, tenta converter; senão mantém string
                if f in sugestoes and sugestoes[f].get("tipo")=="num":
                    try:
                        v = float(v) if "." in v else int(v)
                    except:
                        pass
            entrada[f]=v
        try:
            pred = pipeline.predict(pd.DataFrame([entrada]))[0]
            tarefa = "regressão" if pd.api.types.is_numeric_dtype(df[alvo]) else "classificação"
            resultado = {"task":tarefa, "prediction": round(float(pred),2) if isinstance(pred,(float,int,np.floating)) else str(pred)}
            log_consultation(session.get("user_id"), model_id, alvo, tarefa, entrada, resultado["prediction"])
        except Exception as e:
            resultado={"error": f"Erro ao gerar previsão: {e}"}

    return render_template("analisar.html",
        model_id=model_id, alvo=alvo, feats=feats, sugestoes=sugestoes,
        resultado=resultado, user=current_user(), is_admin=is_admin()
    )


# -------- LOGS ADMIN --------
@app.get("/admin/consultas")
@admin_required
def admin_consultas():
    return render_template("admin_consultas.html", logs=list_consultations(), user=current_user(), is_admin=True)

@app.get("/admin/consultas.csv")
@admin_required
def admin_consultas_csv():
    logs = list_consultations(limit=100000)
    import csv, io
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id","created_at","user_name","user_email","model_id","target","task","input_json","output"])
    for r in logs:
        w.writerow([r["id"],r["created_at"],r["user_name"],r["user_email"],r["model_id"],r["target"],r["task"],r["input_json"],r["output"]])
    out.seek(0)
    return Response(out.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=consultas.csv"})


# -------- ERROS --------
@app.errorhandler(404)
def not_found(e):   return render_template("base.html", message="Página não encontrada."), 404
@app.errorhandler(500)
def server_error(e):return render_template("base.html", message="Erro interno."), 500


# ===================== RUN =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
