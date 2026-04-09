import os, io, base64, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)

app = Flask(__name__)

# ── Accent palette for plots (brutalist) ──────────────────────────────────────
BG   = "#0D0D0D"
FG   = "#F0EBE1"
ACC1 = "#C8F542"   # lime
ACC2 = "#FF4D1C"   # red-orange
ACC3 = "#1CF5FF"   # cyan
GRID = "#1E1E1E"

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(color=GRID, linewidth=0.5)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Train once at startup ──────────────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df_raw = pd.read_csv(CSV)

le = LabelEncoder()
df_raw["label_enc"] = le.fit_transform(df_raw["label"])

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df_raw[FEATURES]
y = df_raw["label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                              criterion="entropy", random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred     = clf.predict(X_test)
train_acc  = clf.score(X_train, y_train)
test_acc   = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred,
                                    target_names=le.classes_, output_dict=True)
cm         = confusion_matrix(y_test, y_pred)

print(f"[CROP.AI] Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    """Dataset overview stats."""
    info = {
        "rows":    int(len(df_raw)),
        "crops":   int(df_raw["label"].nunique()),
        "features": FEATURES,
        "missing": int(df_raw.isnull().sum().sum()),
        "crop_counts": df_raw["label"].value_counts().to_dict(),
        "stats": df_raw[FEATURES].describe().round(2).to_dict(),
    }
    return jsonify(info)


@app.route("/api/model_results")
def api_model_results():
    """Accuracy + classification report table."""
    rows = []
    for crop in le.classes_:
        r = report_dict[crop]
        rows.append({
            "crop":      crop,
            "precision": round(r["precision"], 3),
            "recall":    round(r["recall"],    3),
            "f1":        round(r["f1-score"],  3),
            "support":   int(r["support"]),
        })
    return jsonify({
        "train_accuracy": round(train_acc, 4),
        "test_accuracy":  round(test_acc,  4),
        "rows": rows,
    })


@app.route("/api/plot/actual_vs_predicted")
def plot_avp():
    n = 100
    idx = np.arange(n)
    ya  = np.array(y_test)[:n]
    yp  = np.array(y_pred)[:n]

    fig, ax = plt.subplots(figsize=(11, 4))
    style_ax(ax, fig)
    ax.plot(idx, ya, color=ACC3, lw=1.2, marker="o", ms=3,
            label="Actual", alpha=0.85)
    ax.plot(idx, yp, color=ACC1, lw=1.2, marker="x", ms=4,
            linestyle="--", label="Predicted", alpha=0.85)
    ax.set_title("Actual vs Predicted — first 100 test samples", fontsize=11)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Encoded label")
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor=FG, fontsize=9)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/plot/scatter")
def plot_scatter():
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_test), size=100, replace=False)
    ya  = np.array(y_test)[idx]
    yp  = np.array(y_pred)[idx]

    fig, ax = plt.subplots(figsize=(11, 4))
    style_ax(ax, fig)
    ax.scatter(idx, ya, color=ACC3, alpha=0.75, s=22, marker="o", label="Actual")
    ax.scatter(idx, yp, color=ACC2, alpha=0.75, s=22, marker="x", label="Predicted")
    ax.set_title("Actual vs Predicted — 100 random samples (scatter)", fontsize=11)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Encoded label")
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor=FG, fontsize=9)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/plot/confusion_matrix")
def plot_cm():
    fig, ax = plt.subplots(figsize=(13, 11))
    style_ax(ax, fig)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.3, linecolor="#111",
                cbar_kws={"shrink": 0.7})
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    ax.collections[0].colorbar.ax.tick_params(colors=FG, labelsize=8)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/plot/feature_importance")
def plot_fi():
    imp  = clf.feature_importances_
    sidx = np.argsort(imp)[::-1]
    colors = [ACC1, ACC3, ACC2, "#FF9F1C", "#B388FF", "#80DEEA", "#F48FB1"]

    fig, ax = plt.subplots(figsize=(9, 4))
    style_ax(ax, fig)
    bars = ax.bar(range(len(FEATURES)), imp[sidx],
                  color=[colors[i % len(colors)] for i in range(len(FEATURES))],
                  edgecolor=BG, linewidth=0.8)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in sidx], rotation=25, ha="right")
    ax.set_title("Feature Importances — Random Forest", fontsize=11)
    ax.set_ylabel("Importance score")
    for bar, val in zip(bars, imp[sidx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", color=FG, fontsize=8)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/plot/class_distribution")
def plot_dist():
    counts = df_raw["label"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))

    fig, ax = plt.subplots(figsize=(11, 4))
    style_ax(ax, fig)
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor=BG, lw=0.8)
    ax.set_title("Crop Class Distribution", fontsize=11)
    ax.set_xlabel("Crop"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", color=FG, fontsize=7)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/dataset")
def api_dataset():
    """Return all rows as JSON for the dataset preview table."""
    rows = df_raw[FEATURES + ["label"]].round(2).to_dict(orient="records")
    return jsonify(rows)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        vals = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    arr   = np.array([vals])
    enc   = clf.predict(arr)[0]
    crop  = le.inverse_transform([enc])[0]
    proba = clf.predict_proba(arr)[0]
    top5_idx   = np.argsort(proba)[::-1][:5]
    top5_crops = le.inverse_transform(top5_idx).tolist()
    top5_probs = (proba[top5_idx] * 100).round(1).tolist()

    # Radar chart for input parameters (normalised 0–1)
    maxes = {"N": 150, "P": 160, "K": 250,
             "temperature": 45, "humidity": 100, "ph": 14, "rainfall": 320}
    norm  = [round(vals[i] / maxes[FEATURES[i]], 3) for i in range(len(FEATURES))]

    return jsonify({
        "crop":       crop,
        "top5_crops": top5_crops,
        "top5_probs": top5_probs,
        "norm_vals":  norm,
        "features":   FEATURES,
        "raw_vals":   vals,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
