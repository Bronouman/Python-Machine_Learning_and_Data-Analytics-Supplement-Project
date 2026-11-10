import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def theme_minimal(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fmt_p(p):
    return "< 2.2e-16" if p < 2.2e-16 else f"{p:.4g}"


def tile_plot(mat, x_levels, y_levels_top_to_bottom, xlabel, ylabel, title, label_size):
    cmap = LinearSegmentedColormap.from_list("blues", ["#D6EAF8", "#1A5276"])
    data = mat.loc[y_levels_top_to_bottom, x_levels]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": label_size, "fontweight": "bold", "color": "black"},
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0, rotation=0)
    fig.tight_layout()
    plt.savefig("plot.png")


# ----------------------------------------------------------------
# 1. DATA CLEANING
# ----------------------------------------------------------------
df = pd.read_csv(
    "employee_attrition.csv", dtype=str, keep_default_na=False, na_values=[""]
)

df.columns = df.columns.str.lower()

# --- Categorical Variable: MaritalStatus ---
df["maritalstatus"] = df["maritalstatus"].str.lower()
df["maritalstatus"] = df["maritalstatus"].str.replace(r"[_?]", "", regex=True)
df["maritalstatus"] = df["maritalstatus"].str.strip()

allowed_status = ["married", "divorced", "single"]
df.loc[~df["maritalstatus"].isin(allowed_status), "maritalstatus"] = np.nan

ms_codes = df["maritalstatus"].map({s: i + 1 for i, s in enumerate(allowed_status)})
ms_mean_idx = int(np.round(ms_codes.mean()))
ms_mean_value = allowed_status[ms_mean_idx - 1]
df.loc[df["maritalstatus"].isna(), "maritalstatus"] = ms_mean_value

# --- Numerical Variable: DistanceFromHome ---
df["distancefromhome"] = df["distancefromhome"].str.replace(r"[^0-9]", "", regex=True)
df["distancefromhome"] = pd.to_numeric(df["distancefromhome"], errors="coerce")

df_mean_value = int(np.round(df["distancefromhome"].mean()))
df["distancefromhome"] = df["distancefromhome"].fillna(df_mean_value).astype(int)

# --- Dependent Variable: Attrition ---
df["attrition"] = df["attrition"].replace({"1": "yes", "0": "no"})
df["attrition"] = df["attrition"].str.strip().str.lower()

allowed_attrition = ["yes", "no"]
df.loc[
    ~df["attrition"].isin(allowed_attrition) | df["attrition"].isna(), "attrition"
] = "null"

df = df.reset_index(drop=True)

ATT_MAP = {"no": "Stayed", "yes": "Left"}
ATT_LEVELS = ["Stayed", "Left"]
MS_LEVELS = ["married", "divorced", "single"]


def build(frame, need_ms=True, need_dist=True):
    mask = (frame["attrition"] != "null") & frame["attrition"].notna()
    if need_ms:
        mask &= frame["maritalstatus"].notna()
    if need_dist:
        mask &= frame["distancefromhome"].notna()
    out = frame[mask].copy()
    out["maritalstatus"] = pd.Categorical(out["maritalstatus"], categories=MS_LEVELS)
    out["attrition"] = pd.Categorical(
        out["attrition"].map(ATT_MAP), categories=ATT_LEVELS
    )
    return out.reset_index(drop=True)


# ----------------------------------------------------------------
# ANALYSIS 1: ATTRITION RATE BY MARITALSTATUS
# ----------------------------------------------------------------
df1 = build(df, need_ms=True, need_dist=False)

prop_table1 = (
    pd.crosstab(df1["maritalstatus"], df1["attrition"], normalize="index") * 100
)
prop_table1.index.name = None
prop_table1.columns.name = None
print("\n--- Analysis 1: Proportion Table ---")
print(prop_table1.round(2))

df1_summary = (
    df1.groupby(["maritalstatus", "attrition"], observed=True)
    .size()
    .reset_index(name="count")
)
df1_summary["percentage"] = (
    df1_summary["count"]
    / df1_summary.groupby("maritalstatus", observed=True)["count"].transform("sum")
    * 100
).round(2)

fig, ax = plt.subplots(figsize=(8, 5))
colors1 = {"Stayed": "#2980B9", "Left": "#E74C3C"}
pct_lookup = {
    (r.maritalstatus, r.attrition): r.percentage for r in df1_summary.itertuples()
}
bottoms = {m: 0.0 for m in MS_LEVELS}
for status in ["Left", "Stayed"]:
    for i, m in enumerate(MS_LEVELS):
        val = pct_lookup.get((m, status), 0.0)
        ax.bar(i, val, bottom=bottoms[m], color=colors1[status], width=0.9)
        if val > 0:
            ax.text(
                i,
                bottoms[m] + val / 2,
                f"{val:g}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
        bottoms[m] += val
ax.set_xticks(range(len(MS_LEVELS)))
ax.set_xticklabels(MS_LEVELS)
ax.set_title("Attrition Rate by Marital Status", loc="left", fontweight="bold")
ax.set_xlabel("Marital Status")
ax.set_ylabel("Percentage (%)")
theme_minimal(ax)
ax.legend(
    handles=[Patch(color=colors1[k], label=k) for k in ATT_LEVELS],
    title="Attrition Status",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
)
fig.tight_layout()
plt.savefig("plot.png")

# ----------------------------------------------------------------
# ANALYSIS 2: DISTANCEFROMHOME DISTRIBUTION BY ATTRITION
# ----------------------------------------------------------------
df2 = build(df, need_ms=False, need_dist=True)

df2_stats = (
    df2.groupby("attrition", observed=True)["distancefromhome"]
    .agg(
        mean_dist="mean",
        median_dist="median",
        sd_dist="std",
        var_dist="var",
        iqr_dist=lambda s: s.quantile(0.75) - s.quantile(0.25),
        min_dist="min",
        max_dist="max",
    )
    .reset_index()
)
for c in ["mean_dist", "median_dist", "sd_dist", "var_dist", "iqr_dist"]:
    df2_stats[c] = df2_stats[c].astype(float).round(2)
print("\n--- Analysis 2: Dispersion Statistics ---")
print(df2_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
colors2 = {"Stayed": "#27AE60", "Left": "#E74C3C"}
sns.boxplot(
    data=df2,
    x="attrition",
    y="distancefromhome",
    hue="attrition",
    order=ATT_LEVELS,
    hue_order=ATT_LEVELS,
    palette=colors2,
    legend=False,
    ax=ax,
    boxprops={"alpha": 0.8},
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": 3,
    },
)
for i, r in enumerate(df2_stats.itertuples()):
    ax.annotate(
        f"Mean: {r.mean_dist:g}",
        (i, r.mean_dist),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )
ax.set_title("Distance From Home by Attrition Status", loc="left", fontweight="bold")
ax.set_xlabel("Attrition")
ax.set_ylabel("Distance From Home")
theme_minimal(ax)
ax.legend(
    handles=[Patch(color=colors2[k], label=k) for k in ATT_LEVELS],
    title="Attrition Status",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
)
fig.tight_layout()
plt.savefig("plot.png")

# ----------------------------------------------------------------
# ANALYSIS 3: HYPOTHESIS TESTING (CHI-SQUARE & MANN-WHITNEY U)
# ----------------------------------------------------------------
df3 = build(df, need_ms=True, need_dist=True)

chi_table = pd.crosstab(
    df3["maritalstatus"].cat.remove_unused_categories(),
    df3["attrition"].cat.remove_unused_categories(),
)
chi_table.index.name = None
chi_table.columns.name = None
print("\n--- Analysis 3: Chi-Square Test & Cramer's V ---")
chi2, chi_p, chi_df, _ = stats.chi2_contingency(chi_table.values)
print("\n\tPearson's Chi-squared test\n")
print("data:  chi_table")
print(f"X-squared = {chi2:.5g}, df = {chi_df}, p-value = {fmt_p(chi_p)}\n")

lr, lr_p, _, _ = stats.chi2_contingency(chi_table.values, lambda_="log-likelihood")
n_obs = chi_table.values.sum()
r_dim, c_dim = chi_table.shape
assoc = pd.DataFrame(
    {"X^2": [lr, chi2], "df": [chi_df, chi_df], "P(> X^2)": [lr_p, chi_p]},
    index=["Likelihood Ratio", "Pearson"],
)
print(assoc.round(6).to_string())
phi_txt = f"{np.sqrt(chi2 / n_obs):.3f}" if (r_dim == 2 and c_dim == 2) else "NA"
print(f"\nPhi-Coefficient   : {phi_txt}")
print(f"Contingency Coeff.: {np.sqrt(chi2 / (chi2 + n_obs)):.3f}")
print(f"Cramer's V        : {np.sqrt(chi2 / (n_obs * min(r_dim - 1, c_dim - 1))):.3f}")


mat3 = chi_table.T.copy()
mat3.index = [str(i) for i in mat3.index]
mat3.columns = [str(c) for c in mat3.columns]
tile_plot(
    mat3,
    [m for m in MS_LEVELS if m in mat3.columns],
    ["Left", "Stayed"],
    "Marital Status",
    "Attrition",
    "Association: MaritalStatus vs Attrition",
    14,
)

print("\n--- Analysis 3: Mann-Whitney U Test ---")
x_stayed = df3.loc[df3["attrition"] == "Stayed", "distancefromhome"].to_numpy(
    dtype=float
)
x_left = df3.loc[df3["attrition"] == "Left", "distancefromhome"].to_numpy(dtype=float)
mw = stats.mannwhitneyu(
    x_stayed, x_left, alternative="two-sided", method="asymptotic", use_continuity=True
)
print("\n\tWilcoxon rank sum test with continuity correction\n")
print("data:  distancefromhome by attrition")
print(f"W = {mw.statistic:.5g}, p-value = {fmt_p(mw.pvalue)}")
print("alternative hypothesis: true location shift is not equal to 0\n")


def bw_nrd0(x):
    x = np.asarray(x, dtype=float)
    hi = np.std(x, ddof=1)
    lo = min(hi, (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.34)
    if not lo:
        lo = hi if hi else (abs(x[0]) if abs(x[0]) else 1.0)
    return 0.9 * lo * len(x) ** (-0.2)


fig, ax = plt.subplots(figsize=(8, 5))
colors3 = {"Stayed": "#2980B9", "Left": "#E74C3C"}
grid = np.linspace(df3["distancefromhome"].min(), df3["distancefromhome"].max(), 512)
for k, xs in [("Stayed", x_stayed), ("Left", x_left)]:
    kde = stats.gaussian_kde(xs, bw_method=bw_nrd0(xs) / np.std(xs, ddof=1))
    dens = kde(grid)
    ax.fill_between(grid, dens, color=colors3[k], alpha=0.5, linewidth=0)
    ax.plot(grid, dens, color="black", linewidth=0.6)
ax.set_title("Density of DistanceFromHome by Attrition", loc="left", fontweight="bold")
ax.set_xlabel("Distance From Home")
ax.set_ylabel("Density")
theme_minimal(ax)
ax.legend(
    handles=[
        Patch(facecolor=colors3[k], edgecolor="black", alpha=0.5, label=k)
        for k in ATT_LEVELS
    ],
    title="Status",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
)
fig.tight_layout()
plt.savefig("plot.png")

# ----------------------------------------------------------------
# ANALYSIS 4: COMBINED PATTERN (ANOVA & FACETED VIOLIN)
# ----------------------------------------------------------------
df4 = build(df, need_ms=True, need_dist=True)

df4_summary = (
    df4.groupby(["maritalstatus", "attrition"], observed=True)["distancefromhome"]
    .agg(mean_dist="mean", count="size")
    .reset_index()
)
df4_summary["mean_dist"] = df4_summary["mean_dist"].astype(float).round(2)

print("\n--- Analysis 4: ANOVA Summary ---")
groups = [
    g["distancefromhome"].to_numpy(dtype=float)
    for _, g in df4.groupby("maritalstatus", observed=True)
]
all_vals = np.concatenate(groups)
grand = all_vals.mean()
ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
df_b = len(groups) - 1
df_w = len(all_vals) - len(groups)
ms_b, ms_w = ss_b / df_b, ss_w / df_w
f_val = ms_b / ms_w
f_p = stats.f.sf(f_val, df_b, df_w)
aov_tbl = pd.DataFrame(
    {
        "Df": [df_b, df_w],
        "Sum Sq": [round(ss_b, 2), round(ss_w, 2)],
        "Mean Sq": [round(ms_b, 2), round(ms_w, 2)],
        "F value": [round(f_val, 3), ""],
        "Pr(>F)": [fmt_p(f_p), ""],
    },
    index=["maritalstatus", "Residuals"],
)
print(aov_tbl.to_string())

colors4 = {"married": "#2980B9", "divorced": "#E67E22", "single": "#8E44AD"}
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
for ax, att in zip(axes, ATT_LEVELS):
    sub = df4[df4["attrition"] == att].copy()
    sub["maritalstatus"] = sub["maritalstatus"].astype(str)
    sub["distancefromhome"] = sub["distancefromhome"].astype(float)
    sns.violinplot(
        data=sub,
        x="maritalstatus",
        y="distancefromhome",
        hue="maritalstatus",
        order=MS_LEVELS,
        hue_order=MS_LEVELS,
        palette=colors4,
        cut=3,
        inner=None,
        saturation=1,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    for coll in ax.collections:
        coll.set_alpha(0.6)
    sns.boxplot(
        data=sub,
        x="maritalstatus",
        y="distancefromhome",
        order=MS_LEVELS,
        color="white",
        width=0.15,
        fliersize=2,
        linewidth=0.8,
        ax=ax,
    )
    for r in df4_summary[df4_summary["attrition"] == att].itertuples():
        ax.annotate(
            f"n={r.count}",
            (MS_LEVELS.index(r.maritalstatus), r.mean_dist),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(att)
    ax.set_xlabel("")
    ax.set_ylabel("Distance" if att == "Stayed" else "")
    theme_minimal(ax)
fig.suptitle(
    "DistanceFromHome by MaritalStatus and Attrition",
    x=0.01,
    ha="left",
    fontweight="bold",
)
fig.supxlabel("Marital Status", y=0.115)
fig.legend(
    handles=[Patch(color=colors4[k], alpha=0.6, label=k) for k in MS_LEVELS],
    title="Marital Status",
    loc="lower center",
    ncol=3,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.17, 1, 1))
plt.savefig("plot.png")

# ----------------------------------------------------------------
# ANALYSIS 5: DECISION TREES FOR ATTRITION PREDICTION
# ----------------------------------------------------------------
df5 = build(df, need_ms=True, need_dist=True)
df5["distancefromhome"] = df5["distancefromhome"].astype(float)

train_set, test_set = train_test_split(
    df5, train_size=0.8, stratify=df5["attrition"], random_state=42
)
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)


def rose(data, seed=42):
    """Python implementation of ROSE (smoothed-bootstrap balanced resampling)."""
    rng = np.random.default_rng(seed)
    n_total = len(data)
    new_cls = rng.choice(ATT_LEVELS, size=n_total, p=[0.5, 0.5])
    d_cont = 1
    out_ms = np.empty(n_total, dtype=object)
    out_dist = np.empty(n_total, dtype=float)
    for c in ATT_LEVELS:
        sub = data[data["attrition"] == c]
        idx = np.where(new_cls == c)[0]
        m = len(idx)
        if m == 0:
            continue
        h = (4.0 / ((d_cont + 2) * len(sub))) ** (1.0 / (d_cont + 4)) * sub[
            "distancefromhome"
        ].std(ddof=1)
        seeds = rng.integers(0, len(sub), size=m)
        out_dist[idx] = (
            sub["distancefromhome"].to_numpy()[seeds] + rng.normal(0.0, 1.0, size=m) * h
        )
        out_ms[idx] = rng.choice(
            sub["maritalstatus"].astype(str).to_numpy(), size=m, replace=True
        )
    out = pd.DataFrame(
        {
            "attrition": pd.Categorical(new_cls, categories=ATT_LEVELS),
            "maritalstatus": pd.Categorical(out_ms, categories=MS_LEVELS),
            "distancefromhome": out_dist,
        }
    )
    return out


train_balanced = rose(train_set, seed=42)
print("\n--- Analysis 5: Balanced Training Data ---")
print(
    train_balanced["attrition"]
    .value_counts()
    .reindex(ATT_LEVELS)
    .rename_axis("attrition")
)


def design(d):
    X = pd.DataFrame(
        {
            f"maritalstatus_{lv}": (
                d["maritalstatus"].astype(str).to_numpy() == lv
            ).astype(int)
            for lv in MS_LEVELS
        }
    )
    X["distancefromhome"] = d["distancefromhome"].to_numpy(dtype=float)
    return X


X_train = design(train_balanced)
y_train = (train_balanced["attrition"].astype(str).to_numpy() == "Left").astype(int)
X_test = design(test_set)
y_test = (test_set["attrition"].astype(str).to_numpy() == "Left").astype(int)

dt_model = DecisionTreeClassifier(
    criterion="gini",
    min_samples_split=20,
    min_samples_leaf=7,
    ccp_alpha=0.002,
    random_state=42,
)
dt_model.fit(X_train, y_train)

print("\n--- Analysis 5: Decision Tree Summary ---")
print(
    f"Decision tree (n = {len(X_train)}): depth = {dt_model.get_depth()}, "
    f"leaves = {dt_model.get_n_leaves()}, nodes = {dt_model.tree_.node_count}"
)
imp = pd.Series(dt_model.feature_importances_ * 100, index=X_train.columns).sort_values(
    ascending=False
)
print("\nVariable importance:")
print(imp.round(0).to_string())
print("\nTree rules:")
print(export_text(dt_model, feature_names=list(X_train.columns), show_weights=True))

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    dt_model,
    feature_names=list(X_train.columns),
    class_names=ATT_LEVELS,
    filled=True,
    impurity=False,
    proportion=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree: Employee Attrition", fontweight="bold")
fig.tight_layout()
plt.savefig("plot.png")

pred = dt_model.predict(X_test)
if dt_model.n_classes_ == 2:
    pred_probs = dt_model.predict_proba(X_test)[:, 1]
else:
    pred_probs = np.full(len(X_test), float(dt_model.classes_[0] == 1))

# --- Confusion matrix & caret-style statistics (positive class = Left) ---
cm = confusion_matrix(y_test, pred, labels=[0, 1])
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
n = tn + fp + fn + tp


def sdiv(a, b):
    return a / b if b else float("nan")


acc = (tp + tn) / n
ci = stats.binomtest(tp + tn, n).proportion_ci(confidence_level=0.95, method="exact")
nir = max(tp + fn, tn + fp) / n
acc_p = stats.binomtest(tp + tn, n, p=nir, alternative="greater").pvalue
pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
kappa = sdiv(acc - pe, 1 - pe)
if (fp + fn) > 0:
    mc_stat = (abs(fp - fn) - 1) ** 2 / (fp + fn)
    mc_p = stats.chi2.sf(mc_stat, 1)
else:
    mc_p = float("nan")
sens = sdiv(tp, tp + fn)
spec = sdiv(tn, tn + fp)
ppv = sdiv(tp, tp + fp)
npv = sdiv(tn, tn + fn)
prev = (tp + fn) / n
det_rate = tp / n
det_prev = (tp + fp) / n
bal_acc = (sens + spec) / 2

print("\n--- Analysis 5: Confusion Matrix ---")
print("Confusion Matrix and Statistics\n")
print("          Reference")
print("Prediction Stayed Left")
print(f"    Stayed {tn:>6} {fn:>4}")
print(f"    Left   {fp:>6} {tp:>4}\n")
print(f"               Accuracy : {acc:.4f}")
print(f"                 95% CI : ({ci.low:.4f}, {ci.high:.4f})")
print(f"    No Information Rate : {nir:.4f}")
print(f"    P-Value [Acc > NIR] : {fmt_p(acc_p)}\n")
print(f"                  Kappa : {kappa:.4f}\n")
print(f" Mcnemar's Test P-Value : {fmt_p(mc_p) if not np.isnan(mc_p) else 'NA'}\n")
print(f"            Sensitivity : {sens:.4f}")
print(f"            Specificity : {spec:.4f}")
print(f"         Pos Pred Value : {ppv:.4f}")
print(f"         Neg Pred Value : {npv:.4f}")
print(f"             Prevalence : {prev:.4f}")
print(f"         Detection Rate : {det_rate:.4f}")
print(f"   Detection Prevalence : {det_prev:.4f}")
print(f"      Balanced Accuracy : {bal_acc:.4f}\n")
print("       'Positive' Class : Left\n")

cm_df = pd.DataFrame(
    cm.T, index=ATT_LEVELS, columns=ATT_LEVELS
)  # rows = Predicted, cols = Actual
tile_plot(
    cm_df,
    ATT_LEVELS,
    ["Left", "Stayed"],
    "Actual",
    "Predicted",
    "Confusion Matrix \u2014 Decision Tree Prediction",
    16,
)

auc_val = roc_auc_score(y_test, pred_probs)
print("\n--- Analysis 5: Model AUC ---")
print("AUC:", round(auc_val, 4), "")

fpr, tpr, _ = roc_curve(y_test, pred_probs)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([1, 0], [0, 1], color="darkgrey", linewidth=1)
ax.plot(1 - fpr, tpr, color="#2980B9", linewidth=2)
ax.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=1)
ax.set_xlim(1.0, 0.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Specificity")
ax.set_ylabel("Sensitivity")
ax.set_title(f"ROC Curve | AUC = {round(auc_val, 4)}", fontweight="bold")
fig.tight_layout()
plt.savefig("plot.png")
