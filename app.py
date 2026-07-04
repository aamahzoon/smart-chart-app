# -*- coding: utf-8 -*-
"""
Smart Chart Explorer — Pro
---------------------------
اپلیکیشن استریم‌لیت برای:
  1) تشخیص خودکار مناسب‌ترین نمودار توصیفی از روی ساختار داده
  2) ساخت نمودار دلخواه با کنترل کامل
  3) تحلیل آماری تخصصی زیر هر نمودار (رگرسیون، ANOVA، آزمون نرمالیتی، تحلیل روند زمانی، همبستگی و ...)
     با استفاده از scipy و statsmodels + تولید خودکار بینش‌های تفسیری به زبان فارسی
  4) امکان انتخاب هر نمودار (به‌همراه تحلیل‌هایش) و افزودن آن به یک «گزارش/داشبورد» اختصاصی
     قابل چیدمان و خروجی HTML مستقل

اجرا:
    pip install -r requirements.txt
    streamlit run app.py
"""

import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats as sstats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ============================================================================
# تنظیمات صفحه و ظاهر
# ============================================================================
st.set_page_config(page_title="Smart Chart Explorer Pro", page_icon="📊", layout="wide")

BG_COLOR = "#f4f6fb"          # پس‌زمینه کلی: خاکستری-آبی خیلی روشن
CARD_COLOR = "#ffffff"        # کارت‌ها: سفید
TEXT_COLOR = "#1e293b"        # متن: سرمه‌ای تیره (خوانا روی پس‌زمینه روشن)
ACCENT_COLOR = "#2563eb"      # آبی حرفه‌ای (لهجه اصلی)
ACCENT_2 = "#f59e0b"          # کهربایی شاداب (لهجه دوم / hover)
GRID_COLOR = "#e2e8f0"        # خطوط شبکه: خاکستری روشن
FONT_FAMILY = "Vazirmatn"

PLOTLY_COLORWAY = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
                    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=14),
        title=dict(x=0.98, xanchor="right", font=dict(size=18, color=TEXT_COLOR)),
        colorway=PLOTLY_COLORWAY,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR),
        legend=dict(font=dict(family=FONT_FAMILY), orientation="h", y=-0.25),
        margin=dict(t=60, l=40, r=40, b=40),
    )
)
px.defaults.template = PLOTLY_TEMPLATE
px.defaults.color_discrete_sequence = PLOTLY_COLORWAY

st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
html, body, [class*="css"], .stApp {{
    font-family: '{FONT_FAMILY}', Tahoma, sans-serif !important;
    direction: rtl;
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
}}
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stCaption, .stSelectbox label, .stMultiSelect label,
.stTextInput label, .stSlider label, .stRadio label {{
    direction: rtl; text-align: right;
}}
h1, h2, h3 {{ color: #0f172a; }}
div[data-baseweb="tab-list"] {{ direction: rtl; }}
div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {CARD_COLOR};
    border-radius: 12px;
    border: 1px solid {GRID_COLOR};
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}}
[data-testid="stMetric"] {{
    background-color: {CARD_COLOR};
    border: 1px solid {GRID_COLOR};
    border-radius: 10px;
    padding: 10px 14px;
}}
.stButton > button {{
    background-color: {ACCENT_COLOR}; color: #ffffff; font-weight: 700;
    border-radius: 8px; border: none;
}}
.stButton > button:hover {{ background-color: {ACCENT_2}; color: #1e293b; }}
[data-testid="stMetricValue"] {{ color: {ACCENT_COLOR}; }}
.insight-box {{
    background-color: rgba(37, 99, 235, 0.06);
    border-right: 4px solid {ACCENT_COLOR};
    padding: 12px 16px; border-radius: 8px; margin-top: 8px;
    color: {TEXT_COLOR};
}}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: {GRID_COLOR}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Chart Explorer — نسخه حرفه‌ای")
st.caption("تشخیص خودکار نمودار، تحلیل آماری تخصصی، و ساخت گزارش/داشبورد اختصاصی از نمودارهای منتخب.")


def fa_num(x, decimals=None) -> str:
    """قالب‌بندی عدد با ارقام فارسی."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if decimals is not None:
        x = round(float(x), decimals)
    if isinstance(x, (int, np.integer)):
        text = f"{x:,}"
    elif isinstance(x, float) and x == int(x) and decimals is None:
        text = f"{int(x):,}"
    else:
        text = f"{x:,}" if decimals is None else f"{x:,.{decimals}f}"
    mapping = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    return text.translate(mapping)


# ============================================================================
# بارگذاری و طبقه‌بندی داده
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("فرمت فایل پشتیبانی نمی‌شود. لطفاً CSV یا Excel آپلود کنید.")

    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            except Exception:
                pass
    return df


def classify_columns(df: pd.DataFrame) -> dict:
    numeric_cols, categorical_cols, datetime_cols, boolean_cols = [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 10 and s.dropna().shape[0] > 0:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {"numeric": numeric_cols, "categorical": categorical_cols,
            "datetime": datetime_cols, "boolean": boolean_cols}


# ============================================================================
# موتور تحلیل آماری (scipy + statsmodels) و تولید بینش فارسی
# ============================================================================
def strength_label(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "ناچیز"
    if a < 0.3:
        return "ضعیف"
    if a < 0.5:
        return "متوسط"
    if a < 0.7:
        return "قوی"
    return "بسیار قوی"


def analyze_distribution(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna().astype(float)
    n = len(s)
    if n < 3:
        return {"error": "داده کافی برای تحلیل توزیع وجود ندارد."}

    mean, median, std = s.mean(), s.median(), s.std()
    skew, kurt = sstats.skew(s), sstats.kurtosis(s)

    if n <= 5000:
        stat, p = sstats.shapiro(s)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = sstats.normaltest(s)
        test_name = "D'Agostino K²"

    is_normal = p > 0.05
    skew_desc = "متقارن تقریباً" if abs(skew) < 0.5 else ("راست‌چوله (دنباله بلند به سمت مقادیر بزرگ)" if skew > 0 else "چپ‌چوله (دنباله بلند به سمت مقادیر کوچک)")

    close_center = bool(std) and abs(mean - median) < 0.1 * std
    center_desc = (
        "میانگین و میانه نزدیک‌اند که نشانهٔ توزیع نسبتاً متقارن است."
        if close_center else
        "اختلاف میانگین و میانه نشان‌دهندهٔ وجود چولگی در توزیع است."
    )

    return {
        "n": n, "mean": mean, "median": median, "std": std,
        "min": s.min(), "max": s.max(), "cv": (std / mean * 100) if mean else np.nan,
        "skew": skew, "kurt": kurt, "test_name": test_name, "stat": stat, "p": p,
        "is_normal": is_normal, "skew_desc": skew_desc,
        "insight": (
            f"- میانگین **{fa_num(mean, 2)}** و میانه **{fa_num(median, 2)}** است؛ {center_desc}\n"
            f"- شکل توزیع: **{skew_desc}** (ضریب چولگی = {fa_num(skew,2)}).\n"
            f"- ضریب تغییرات (پراکندگی نسبی) برابر **{fa_num((std/mean*100) if mean else float('nan'),1)}٪** است.\n"
            f"- بر اساس آزمون **{test_name}** (p-value = {fa_num(p,4)})، "
            f"{'نمی‌توان فرض نرمال بودن توزیع را رد کرد.' if is_normal else 'توزیع به‌طور معناداری از توزیع نرمال فاصله دارد.'}"
        ),
    }


def analyze_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    d = df[[x_col, y_col]].dropna()
    d = d[pd.to_numeric(d[x_col], errors="coerce").notna() & pd.to_numeric(d[y_col], errors="coerce").notna()]
    if len(d) < 3:
        return {"error": "داده کافی برای تحلیل رگرسیون وجود ندارد."}

    x = d[x_col].astype(float).values
    y = d[y_col].astype(float).values
    r, p_r = sstats.pearsonr(x, y)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params[1]
    intercept = model.params[0]
    slope_p = model.pvalues[1]
    r2 = model.rsquared

    direction = "مثبت (هم‌جهت)" if slope > 0 else "منفی (خلاف‌جهت)"
    significant = slope_p < 0.05

    return {
        "n": len(d), "r": r, "p_r": p_r, "slope": slope, "intercept": intercept,
        "slope_p": slope_p, "r2": r2, "direction": direction, "significant": significant,
        "insight": (
            f"- همبستگی پیرسون بین {x_col} و {y_col}: **r = {fa_num(r,3)}** "
            f"(همبستگی **{strength_label(r)}** و **{direction}**).\n"
            f"- معادلهٔ خط برازش‌شده: y ≈ {fa_num(intercept,3)} + {fa_num(slope,4)} × x\n"
            f"- ضریب تعیین **R² = {fa_num(r2*100,1)}٪** است؛ یعنی این درصد از تغییرات {y_col} "
            f"توسط {x_col} توضیح داده می‌شود.\n"
            f"- شیب رگرسیون از نظر آماری **{'معنادار است (p = ' + fa_num(slope_p,4) + ')' if significant else 'معنادار نیست (p = ' + fa_num(slope_p,4) + ')'}**؛ "
            f"{'یعنی رابطهٔ مشاهده‌شده تصادفی نیست.' if significant else 'یعنی نمی‌توان با اطمینان از وجود رابطهٔ خطی صحبت کرد.'}"
        ),
    }


def analyze_anova(df: pd.DataFrame, num_col: str, cat_col: str) -> dict:
    d = df[[num_col, cat_col]].dropna()
    groups_dict = {k: v[num_col].astype(float).values for k, v in d.groupby(cat_col) if len(v) >= 2}
    groups = list(groups_dict.values())
    if len(groups) < 2:
        return {"error": "برای مقایسه بین‌گروهی حداقل به دو گروه با داده کافی نیاز است."}

    f_stat, p_val = sstats.f_oneway(*groups)
    try:
        levene_stat, levene_p = sstats.levene(*groups)
    except Exception:
        levene_stat, levene_p = np.nan, np.nan

    means = d.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
    top_group, bottom_group = means.index[0], means.index[-1]
    significant = p_val < 0.05

    return {
        "f_stat": f_stat, "p_val": p_val, "means": means,
        "levene_p": levene_p, "significant": significant,
        "top_group": top_group, "bottom_group": bottom_group,
        "insight": (
            f"- بر اساس تحلیل واریانس (ANOVA) با آماره **F = {fa_num(f_stat,2)}** و **p-value = {fa_num(p_val,4)}**، "
            f"{'اختلاف میانگین‌ها بین گروه‌های ' + cat_col + ' از نظر آماری معنادار است.' if significant else 'اختلاف میانگین‌ها بین گروه‌ها معنادار نیست و می‌تواند ناشی از نوسان تصادفی باشد.'}\n"
            f"- بیشترین میانگین {num_col} مربوط به گروه **«{top_group}»** ({fa_num(means.iloc[0],2)}) "
            f"و کمترین مربوط به گروه **«{bottom_group}»** ({fa_num(means.iloc[-1],2)}) است.\n"
            f"- آزمون Levene برای همگنی واریانس‌ها: p-value = {fa_num(levene_p,4)} "
            f"({'واریانس گروه‌ها تقریباً برابرند' if (not np.isnan(levene_p) and levene_p > 0.05) else 'ممکن است واریانس گروه‌ها نابرابر باشد؛ نتایج ANOVA را با احتیاط تفسیر کنید'})."
        ),
    }


def analyze_timeseries(df: pd.DataFrame, date_col: str, num_col: str) -> dict:
    d = df[[date_col, num_col]].dropna().sort_values(date_col)
    if len(d) < 5:
        return {"error": "داده کافی برای تحلیل روند زمانی وجود ندارد."}

    t = np.arange(len(d))
    y = d[num_col].astype(float).values
    X = sm.add_constant(t)
    model = sm.OLS(y, X).fit()
    slope, slope_p, r2 = model.params[1], model.pvalues[1], model.rsquared

    try:
        adf_stat, adf_p = adfuller(y, autolag="AIC")[:2]
        is_stationary = adf_p < 0.05
    except Exception:
        adf_stat, adf_p, is_stationary = np.nan, np.nan, None

    trend_desc = "افزایشی" if slope > 0 else "کاهشی"
    significant = slope_p < 0.05

    return {
        "n": len(d), "slope": slope, "slope_p": slope_p, "r2": r2,
        "adf_p": adf_p, "is_stationary": is_stationary, "trend_desc": trend_desc,
        "significant": significant,
        "insight": (
            f"- روند کلی سری زمانی **{trend_desc}** است "
            f"({'از نظر آماری معنادار، p = ' + fa_num(slope_p,4) if significant else 'اما از نظر آماری معنادار نیست، p = ' + fa_num(slope_p,4)}).\n"
            f"- به‌طور میانگین، مقدار {num_col} در هر گام زمانی حدود **{fa_num(slope,3)}** واحد تغییر می‌کند.\n"
            f"- برازش خط روند {fa_num(r2*100,1)}٪ از تغییرات را توضیح می‌دهد (R²).\n"
            + (f"- آزمون ایستایی ADF: p = {fa_num(adf_p,4)} → سری زمانی "
               f"{'ایستا (Stationary) است' if is_stationary else 'ایستا نیست و احتمالاً دارای روند یا ساختار زمانی است'}."
               if is_stationary is not None else "")
        ),
    }


def analyze_categorical(df: pd.DataFrame, col: str) -> dict:
    vc = df[col].value_counts(dropna=True)
    if len(vc) < 2:
        return {"error": "برای این تحلیل حداقل به دو دسته نیاز است."}

    n = vc.sum()
    top_cat, top_share = vc.index[0], vc.iloc[0] / n * 100
    hhi = float(((vc / n) ** 2).sum())  # شاخص تمرکز هرفیندال (بین ۱/تعداد دسته تا ۱)

    expected = np.full(len(vc), n / len(vc))
    chi2, p_val = sstats.chisquare(vc.values, f_exp=expected)
    is_uniform = p_val > 0.05

    return {
        "n_categories": len(vc), "top_cat": top_cat, "top_share": top_share,
        "hhi": hhi, "chi2": chi2, "p_val": p_val, "is_uniform": is_uniform,
        "insight": (
            f"- پرتکرارترین دسته **«{top_cat}»** با سهم **{fa_num(top_share,1)}٪** از کل مشاهدات است.\n"
            f"- شاخص تمرکز هرفیندال (HHI) برابر **{fa_num(hhi,3)}** است؛ "
            f"{'توزیع نسبتاً متمرکز روی چند دسته است.' if hhi > 0.25 else 'توزیع بین دسته‌ها نسبتاً پراکنده است.'}\n"
            f"- آزمون Chi-Square برای یکنواختی توزیع: p-value = {fa_num(p_val,4)} → "
            f"{'توزیع دسته‌ها را نمی‌توان به‌طور معناداری غیریکنواخت دانست.' if is_uniform else 'توزیع دسته‌ها به‌طور معناداری غیریکنواخت است (برخی دسته‌ها به‌وضوح غالب‌ترند).'}"
        ),
    }


def analyze_correlation(df: pd.DataFrame, numeric_cols: list) -> dict:
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            r = corr.loc[c1, c2]
            if pd.notna(r):
                pairs.append((c1, c2, r))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    if not pairs:
        return {"error": "داده کافی برای محاسبه همبستگی وجود ندارد."}

    top_pairs = pairs[:3]
    lines = [
        f"- **{c1} ↔ {c2}**: r = {fa_num(r,3)} (همبستگی {strength_label(r)}، {'مثبت' if r > 0 else 'منفی'})"
        for c1, c2, r in top_pairs
    ]
    return {
        "corr": corr, "top_pairs": top_pairs,
        "insight": "قوی‌ترین رابطه‌های همبستگی مشاهده‌شده:\n" + "\n".join(lines),
    }


# ============================================================================
# رندر بلوک تحلیل آماری + دکمهٔ افزودن به گزارش
# ============================================================================
if "report_items" not in st.session_state:
    st.session_state.report_items = []


def render_metrics(metric_pairs):
    cols = st.columns(len(metric_pairs))
    for c, (label, value) in zip(cols, metric_pairs):
        c.metric(label, value)


def render_analysis_block(unique_key: str, title: str, fig, analysis_type: str, analysis: dict):
    """نمایش نمودار + تحلیل آماری + دکمه افزودن به گزارش."""
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key}")

    with st.expander("📈 تحلیل آماری تخصصی و بینش‌ها", expanded=False):
        if "error" in analysis:
            st.info(analysis["error"])
        else:
            if analysis_type == "distribution":
                render_metrics([
                    ("میانگین", fa_num(analysis["mean"], 2)),
                    ("میانه", fa_num(analysis["median"], 2)),
                    ("انحراف معیار", fa_num(analysis["std"], 2)),
                    (f"آزمون {analysis['test_name']} (p)", fa_num(analysis["p"], 4)),
                ])
            elif analysis_type == "regression":
                render_metrics([
                    ("ضریب همبستگی (r)", fa_num(analysis["r"], 3)),
                    ("ضریب تعیین (R²)", f"{fa_num(analysis['r2']*100,1)}٪"),
                    ("شیب خط", fa_num(analysis["slope"], 4)),
                    ("معناداری شیب (p)", fa_num(analysis["slope_p"], 4)),
                ])
            elif analysis_type == "anova":
                render_metrics([
                    ("آماره F", fa_num(analysis["f_stat"], 2)),
                    ("p-value", fa_num(analysis["p_val"], 4)),
                    ("بیشترین میانگین", str(analysis["top_group"])),
                    ("کمترین میانگین", str(analysis["bottom_group"])),
                ])
            elif analysis_type == "timeseries":
                render_metrics([
                    ("روند", analysis["trend_desc"]),
                    ("R² خط روند", f"{fa_num(analysis['r2']*100,1)}٪"),
                    ("معناداری روند (p)", fa_num(analysis["slope_p"], 4)),
                    ("وضعیت ایستایی", "ایستا" if analysis.get("is_stationary") else "ناایستا" if analysis.get("is_stationary") is not None else "—"),
                ])
            elif analysis_type == "categorical":
                render_metrics([
                    ("تعداد دسته‌ها", fa_num(analysis["n_categories"])),
                    ("سهم دستهٔ غالب", f"{fa_num(analysis['top_share'],1)}٪"),
                    ("شاخص تمرکز (HHI)", fa_num(analysis["hhi"], 3)),
                    ("یکنواختی (p-value)", fa_num(analysis["p_val"], 4)),
                ])
            elif analysis_type == "correlation" and "top_pairs" in analysis:
                for c1, c2, r in analysis["top_pairs"]:
                    st.metric(f"{c1} ↔ {c2}", fa_num(r, 3))

            st.markdown(f'<div class="insight-box">💡 <b>بینش‌های تحلیلی:</b><br>{analysis["insight"]}</div>',
                        unsafe_allow_html=True)

    if st.button("➕ افزودن این نمودار و تحلیل به گزارش من", key=f"add_{unique_key}"):
        stats_html = ""
        if "error" not in analysis:
            insight_html = analysis["insight"].replace("\n", "<br>")
        else:
            insight_html = analysis["error"]
        st.session_state.report_items.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "fig_html": (
                '<div style="width:100%;height:460px;">'
                + pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False})
                + '</div>'
            ),
            "insight_html": insight_html,
            "width": "full",
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        st.success(f"✅ نمودار «{title}» به گزارش اضافه شد. برای مشاهده به تب «گزارش و داشبورد من» برو.")


# ============================================================================
# موتور پیشنهاد نمودار خودکار (برمی‌گرداند: عنوان، فیگور، نوع تحلیل، پارامترها)
# ============================================================================
def suggest_charts(df: pd.DataFrame, cols: dict) -> list:
    suggestions = []
    numeric = cols["numeric"]
    categorical = cols["categorical"]
    datetime_cols = cols["datetime"]
    n_rows = len(df)

    if datetime_cols and numeric:
        dcol = datetime_cols[0]
        for ncol in numeric[:2]:
            df_sorted = df.sort_values(dcol)
            fig = px.line(df_sorted, x=dcol, y=ncol, markers=True, title=f"روند {ncol} در طول زمان ({dcol})")
            suggestions.append({"title": f"روند زمانی: {ncol} بر حسب {dcol}", "fig": fig,
                                 "type": "timeseries", "params": {"date_col": dcol, "num_col": ncol}})

    if numeric:
        ncol = numeric[0]
        fig = px.histogram(df, x=ncol, nbins=30, marginal="box", title=f"توزیع {ncol}")
        suggestions.append({"title": f"توزیع (هیستوگرام) {ncol}", "fig": fig,
                             "type": "distribution", "params": {"col": ncol}})

    if len(numeric) >= 2:
        x, y = numeric[0], numeric[1]
        color = categorical[0] if categorical else None
        fig = px.scatter(df, x=x, y=y, color=color, trendline="ols" if n_rows < 5000 else None,
                          title=f"رابطهٔ {x} و {y}")
        suggestions.append({"title": f"پراکنش: {x} در برابر {y}", "fig": fig,
                             "type": "regression", "params": {"x_col": x, "y_col": y}})

    if categorical:
        ccol = categorical[0]
        vc = df[ccol].value_counts(dropna=False).reset_index()
        vc.columns = [ccol, "تعداد"]
        vc_top = vc.head(20)
        fig = px.bar(vc_top, x=ccol, y="تعداد", title=f"فراوانی مقادیر {ccol}")
        suggestions.append({"title": f"فراوانی دسته‌ها: {ccol}", "fig": fig,
                             "type": "categorical", "params": {"col": ccol}})

        if df[ccol].nunique(dropna=True) <= 8:
            fig_pie = px.pie(vc_top, names=ccol, values="تعداد", title=f"سهم هر دسته در {ccol}")
            suggestions.append({"title": f"نمودار دایره‌ای: {ccol}", "fig": fig_pie,
                                 "type": "categorical", "params": {"col": ccol}})

    if numeric and categorical:
        ncol, ccol = numeric[0], categorical[0]
        if df[ccol].nunique(dropna=True) <= 15:
            fig = px.box(df, x=ccol, y=ncol, color=ccol, title=f"مقایسهٔ توزیع {ncol} بر اساس {ccol}")
            suggestions.append({"title": f"مقایسه دسته‌ای: {ncol} بر اساس {ccol}", "fig": fig,
                                 "type": "anova", "params": {"num_col": ncol, "cat_col": ccol}})

    if len(numeric) >= 3:
        corr = df[numeric].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="ماتریس همبستگی متغیرهای عددی")
        suggestions.append({"title": "ماتریس همبستگی", "fig": fig,
                             "type": "correlation", "params": {"numeric_cols": numeric}})

    return suggestions


def run_analysis(item: dict, df: pd.DataFrame) -> dict:
    t, p = item["type"], item["params"]
    if t == "distribution":
        return analyze_distribution(df, p["col"])
    if t == "regression":
        return analyze_regression(df, p["x_col"], p["y_col"])
    if t == "anova":
        return analyze_anova(df, p["num_col"], p["cat_col"])
    if t == "timeseries":
        return analyze_timeseries(df, p["date_col"], p["num_col"])
    if t == "categorical":
        return analyze_categorical(df, p["col"])
    if t == "correlation":
        return analyze_correlation(df, p["numeric_cols"])
    return {"error": "نوع تحلیل نامشخص است."}


# ============================================================================
# رابط کاربری اصلی
# ============================================================================
uploaded_file = st.file_uploader("فایل داده را آپلود کنید (CSV یا Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("برای شروع، یک فایل CSV یا Excel آپلود کنید.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"خطا در خواندن فایل: {e}")
    st.stop()

if df.empty:
    st.warning("فایل آپلودشده خالی است.")
    st.stop()

cols = classify_columns(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("تعداد ردیف‌ها", fa_num(len(df)))
m2.metric("تعداد ستون‌ها", fa_num(df.shape[1]))
m3.metric("ستون‌های عددی", fa_num(len(cols["numeric"])))
m4.metric("ستون‌های دسته‌ای", fa_num(len(cols["categorical"])))

with st.expander("🔍 پیش‌نمایش داده", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown(f"**ستون‌های عددی:** {', '.join(cols['numeric']) or '—'}")
    st.markdown(f"**ستون‌های دسته‌ای:** {', '.join(cols['categorical']) or '—'}")
    st.markdown(f"**ستون‌های تاریخی:** {', '.join(cols['datetime']) or '—'}")
    st.markdown(f"**ستون‌های بولی:** {', '.join(cols['boolean']) or '—'}")

tab_auto, tab_manual, tab_report = st.tabs(
    ["🤖 نمودار پیشنهادی خودکار", "🛠️ نمودار دلخواه من", f"📑 گزارش و داشبورد من ({len(st.session_state.report_items)})"]
)

# --------------------------- تب ۱: تشخیص خودکار -----------------------------
with tab_auto:
    st.subheader("نمودارهای پیشنهادشده بر اساس ساختار داده")
    suggestions = suggest_charts(df, cols)
    st.caption(f"{fa_num(len(suggestions))} نمودار همراه با تحلیل آماری تخصصی پیشنهاد شد.")

    for i, item in enumerate(suggestions):
        st.markdown(f"### {item['title']}")
        analysis = run_analysis(item, df)
        render_analysis_block(f"auto_{i}", item["title"], item["fig"], item["type"], analysis)
        st.divider()

# --------------------------- تب ۲: نمودار دلخواه -----------------------------
with tab_manual:
    st.subheader("نمودار خودت رو بساز")

    chart_type = st.selectbox(
        "نوع نمودار",
        ["خطی (Line)", "میله‌ای (Bar)", "پراکنش (Scatter)", "هیستوگرام",
         "باکس‌پلات (Box)", "ویولن (Violin)", "دایره‌ای (Pie)", "نقشه حرارتی همبستگی (Heatmap)"],
    )

    all_cols = list(df.columns)
    numeric_cols_ext = cols["numeric"] + cols["boolean"]
    x_col = y_col = color_col = None
    agg_func = None
    bins = 30
    names_col = values_col = None
    selected_numeric = []

    if chart_type in ["خطی (Line)", "میله‌ای (Bar)", "پراکنش (Scatter)", "باکس‌پلات (Box)", "ویولن (Violin)"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("محور X", all_cols, key="x")
        with c2:
            y_options = [c for c in all_cols if c != x_col]
            y_col = st.selectbox("محور Y", y_options, key="y")
        with c3:
            color_options = ["— هیچکدام —"] + [c for c in all_cols if c not in (x_col, y_col)]
            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)", color_options, key="color")
            color_col = None if color_col == "— هیچکدام —" else color_col

        if chart_type in ["میله‌ای (Bar)", "خطی (Line)"]:
            agg_func = st.selectbox("نحوهٔ تجمیع مقادیر Y در صورت تکرار X",
                                     ["بدون تجمیع (مقادیر خام)", "میانگین", "مجموع", "تعداد", "میانه"])

    elif chart_type == "هیستوگرام":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("ستون برای بررسی توزیع", all_cols, key="hist_x")
        with c2:
            color_options = ["— هیچکدام —"] + [c for c in all_cols if c != x_col]
            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)", color_options, key="hist_color")
            color_col = None if color_col == "— هیچکدام —" else color_col
        bins = st.slider("تعداد بازه‌ها (bins)", 5, 100, 30)

    elif chart_type == "دایره‌ای (Pie)":
        c1, c2 = st.columns(2)
        with c1:
            names_col = st.selectbox("ستون دسته‌بندی (برچسب‌ها)", all_cols, key="pie_names")
        with c2:
            value_options = ["تعداد (شمارش خودکار)"] + numeric_cols_ext
            values_col = st.selectbox("ستون مقدار", value_options, key="pie_values")

    elif chart_type == "نقشه حرارتی همبستگی (Heatmap)":
        selected_numeric = st.multiselect("ستون‌های عددی برای محاسبه همبستگی",
                                           numeric_cols_ext, default=numeric_cols_ext[:6])

    plot_title = st.text_input("عنوان نمودار (اختیاری)", value="")

    if st.button("رسم نمودار", type="primary"):
        try:
            fig = None
            analysis_type = None
            analysis_params = {}

            if chart_type in ["خطی (Line)", "میله‌ای (Bar)"] and agg_func and agg_func != "بدون تجمیع (مقادیر خام)":
                agg_map = {"میانگین": "mean", "مجموع": "sum", "تعداد": "count", "میانه": "median"}
                group_cols = [x_col] + ([color_col] if color_col else [])
                plot_df = df.groupby(group_cols, dropna=False)[y_col].agg(agg_map[agg_func]).reset_index()
            else:
                plot_df = df

            if chart_type == "خطی (Line)":
                fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, markers=True)
                if pd.api.types.is_datetime64_any_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    analysis_type, analysis_params = "timeseries", {"date_col": x_col, "num_col": y_col}
                elif pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    analysis_type, analysis_params = "regression", {"x_col": x_col, "y_col": y_col}

            elif chart_type == "میله‌ای (Bar)":
                fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, barmode="group")
                if pd.api.types.is_numeric_dtype(df[y_col]) and not pd.api.types.is_numeric_dtype(df[x_col]):
                    analysis_type, analysis_params = "anova", {"num_col": y_col, "cat_col": x_col}

            elif chart_type == "پراکنش (Scatter)":
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, trendline="ols")
                analysis_type, analysis_params = "regression", {"x_col": x_col, "y_col": y_col}

            elif chart_type == "هیستوگرام":
                fig = px.histogram(plot_df, x=x_col, color=color_col, nbins=bins, marginal="box")
                analysis_type, analysis_params = "distribution", {"col": x_col}

            elif chart_type == "باکس‌پلات (Box)":
                fig = px.box(plot_df, x=x_col, y=y_col, color=color_col)
                analysis_type, analysis_params = "anova", {"num_col": y_col, "cat_col": x_col}

            elif chart_type == "ویولن (Violin)":
                fig = px.violin(plot_df, x=x_col, y=y_col, color=color_col, box=True)
                analysis_type, analysis_params = "anova", {"num_col": y_col, "cat_col": x_col}

            elif chart_type == "دایره‌ای (Pie)":
                if values_col == "تعداد (شمارش خودکار)":
                    vc = df[names_col].value_counts(dropna=False).reset_index()
                    vc.columns = [names_col, "تعداد"]
                    fig = px.pie(vc, names=names_col, values="تعداد")
                else:
                    fig = px.pie(df, names=names_col, values=values_col)
                analysis_type, analysis_params = "categorical", {"col": names_col}

            elif chart_type == "نقشه حرارتی همبستگی (Heatmap)":
                if len(selected_numeric) < 2:
                    st.warning("حداقل دو ستون عددی برای محاسبه همبستگی انتخاب کنید.")
                else:
                    corr = df[selected_numeric].corr(numeric_only=True)
                    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    analysis_type, analysis_params = "correlation", {"numeric_cols": selected_numeric}

            if fig is not None:
                if plot_title:
                    fig.update_layout(title=plot_title)
                st.session_state["_manual_fig"] = fig
                st.session_state["_manual_analysis_type"] = analysis_type
                st.session_state["_manual_analysis_params"] = analysis_params
                st.session_state["_manual_title"] = plot_title or chart_type
        except Exception as e:
            st.error(f"خطا در رسم نمودار: {e}")

    if st.session_state.get("_manual_fig") is not None:
        atype = st.session_state.get("_manual_analysis_type")
        aparams = st.session_state.get("_manual_analysis_params") or {}
        if atype:
            analysis = run_analysis({"type": atype, "params": aparams}, df)
        else:
            analysis = {"error": "برای این ترکیب نمودار، تحلیل آماری خودکار در دسترس نیست."}
        render_analysis_block("manual_current", st.session_state.get("_manual_title", "نمودار دلخواه"),
                               st.session_state["_manual_fig"], atype or "none", analysis)

# --------------------------- تب ۳: گزارش و داشبورد من -----------------------
with tab_report:
    st.subheader("📑 گزارش و داشبورد اختصاصی من")

    if not st.session_state.report_items:
        st.info("هنوز نموداری به گزارش اضافه نکرده‌ای. از تب‌های «خودکار» یا «دلخواه» روی «➕ افزودن به گزارش» بزن.")
    else:
        report_title = st.text_input("عنوان گزارش", value="گزارش تحلیلی داده")
        report_desc = st.text_area("توضیح مختصر گزارش (اختیاری)", value="")

        st.caption(f"تعداد نمودارهای گزارش: {fa_num(len(st.session_state.report_items))} — می‌توانی ترتیب و چیدمان را تغییر دهی.")

        items = st.session_state.report_items
        for idx, item in enumerate(items):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{idx+1}. {item['title']}**  \n<small>افزوده‌شده: {item['added_at']}</small>", unsafe_allow_html=True)
                with c2:
                    width_choice = st.selectbox("چیدمان", ["تمام عرض", "نصف عرض"],
                                                 index=0 if item["width"] == "full" else 1,
                                                 key=f"width_{item['id']}")
                    item["width"] = "full" if width_choice == "تمام عرض" else "half"

                bcol1, bcol2, bcol3, _ = st.columns([1, 1, 1, 5])
                with bcol1:
                    if idx > 0 and st.button("⬆️ بالا", key=f"up_{item['id']}"):
                        items[idx - 1], items[idx] = items[idx], items[idx - 1]
                        st.rerun()
                with bcol2:
                    if idx < len(items) - 1 and st.button("⬇️ پایین", key=f"down_{item['id']}"):
                        items[idx + 1], items[idx] = items[idx], items[idx + 1]
                        st.rerun()
                with bcol3:
                    if st.button("🗑️ حذف", key=f"del_{item['id']}"):
                        st.session_state.report_items = [x for x in items if x["id"] != item["id"]]
                        st.rerun()

                with st.expander("پیش‌نمایش", expanded=False):
                    st.components.v1.html(
                        f"<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>{item['fig_html']}",
                        height=480, scrolling=True,
                    )
                    st.markdown(f'<div class="insight-box">{item["insight_html"]}</div>', unsafe_allow_html=True)

        st.divider()
        col_clear, col_export = st.columns([1, 1])
        with col_clear:
            if st.button("🧹 پاک‌کردن کل گزارش"):
                st.session_state.report_items = []
                st.rerun()

        with col_export:
            def build_report_html(items, title, desc):
                blocks = []
                i = 0
                while i < len(items):
                    it = items[i]
                    if it["width"] == "half" and i + 1 < len(items) and items[i + 1]["width"] == "half":
                        it2 = items[i + 1]
                        blocks.append(f"""
                        <div class="row">
                          <div class="card half">
                            <h3>{it['title']}</h3>{it['fig_html']}
                            <div class="insight">{it['insight_html']}</div>
                          </div>
                          <div class="card half">
                            <h3>{it2['title']}</h3>{it2['fig_html']}
                            <div class="insight">{it2['insight_html']}</div>
                          </div>
                        </div>""")
                        i += 2
                    else:
                        blocks.append(f"""
                        <div class="row">
                          <div class="card full">
                            <h3>{it['title']}</h3>{it['fig_html']}
                            <div class="insight">{it['insight_html']}</div>
                          </div>
                        </div>""")
                        i += 1

                html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
* {{ box-sizing: border-box; }}
body {{ font-family: 'Vazirmatn', Tahoma, sans-serif; background:{BG_COLOR}; color:{TEXT_COLOR}; margin:0; padding:32px; direction: rtl; }}
h1 {{ color:#0f172a; margin-bottom:4px; }}
.desc {{ color:#64748b; margin-bottom:28px; }}
.row {{ display:flex; gap:18px; margin-bottom:22px; flex-wrap: wrap; }}
.card {{
    background:{CARD_COLOR}; border:1px solid {GRID_COLOR}; border-radius:14px; padding:18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}}
.card h3 {{ margin-top:0; color:#0f172a; border-right:4px solid {ACCENT_COLOR}; padding-right:10px; }}
.card.full {{ flex: 1 1 100%; }}
.card.half {{ flex: 1 1 calc(50% - 9px); min-width: 320px; }}
.insight {{ background: rgba(37, 99, 235, 0.06); border-right:4px solid {ACCENT_2}; padding:10px 14px; border-radius:8px; margin-top:10px; line-height: 1.9; }}
footer {{ margin-top:32px; color:#94a3b8; font-size:13px; text-align:center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="desc">{desc}</p>
{''.join(blocks)}
<footer>ساخته‌شده با Smart Chart Explorer Pro · {datetime.now().strftime('%Y-%m-%d %H:%M')}</footer>
</body>
</html>"""
                return html

            report_html = build_report_html(st.session_state.report_items, report_title, report_desc)
            st.download_button(
                "📥 دانلود گزارش به‌صورت HTML مستقل",
                data=report_html.encode("utf-8"),
                file_name=f"{report_title.replace(' ', '_') or 'report'}.html",
                mime="text/html",
                type="primary",
            )

st.divider()
st.caption("ساخته‌شده با Streamlit · Pandas · NumPy · Plotly · SciPy · Statsmodels")
