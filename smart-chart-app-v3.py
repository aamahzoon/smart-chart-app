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
st.caption("تشخیص خودکار نمودار، تحلیل آماری تخصصی، و ساخت گزارش/داشبورد اختصاصی از نمودارهای منتخب. (📉 طراحی و ساخت: علی اکبر محزون)")


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


def render_analysis_block(unique_key: str, title: str, fig, analysis_type: str, analysis: dict, is_html: bool = False):
    """نمایش نمودار (یا محتوای HTML مثل کارت‌های اطلاعاتی) + تحلیل آماری + دکمه افزودن به گزارش."""
    if is_html:
        st.markdown(fig, unsafe_allow_html=True)
    else:
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
        if is_html:
            fig_html_content = f'<div style="width:100%;">{fig}</div>'
        else:
            fig_html_content = (
                '<div style="width:100%;height:460px;">'
                + pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False})
                + '</div>'
            )
        st.session_state.report_items.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "fig_html": fig_html_content,
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
# فیلتر و برش سراسری داده (Slicers)
# ============================================================================
def render_filter_panel(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """پنل فیلتر/برش سراسری؛ روی همه نمودارهای برنامه (خودکار و دلخواه) اعمال می‌شود."""
    with st.expander("🔍 فیلتر و برش داده (Filters & Slicers)", expanded=False):
        filterable_cols = cols["categorical"] + cols["numeric"] + cols["datetime"] + cols["boolean"]
        chosen = st.multiselect(
            "ستون‌هایی که می‌خواهی روی آن‌ها فیلتر/برش بگذاری",
            filterable_cols, key="filter_cols_choice",
        )
        filtered = df.copy()
        if not chosen:
            st.caption("برای افزودن فیلتر یا برش، یک یا چند ستون از لیست بالا انتخاب کن. این فیلتر روی همهٔ نمودارهای برنامه (تب خودکار و تب دلخواه) اعمال می‌شود.")
            return filtered

        for col in chosen:
            if col in cols["datetime"]:
                s = df[col].dropna()
                if s.empty:
                    continue
                min_d, max_d = s.min().date(), s.max().date()
                dr = st.date_input(f"بازهٔ تاریخ «{col}»", value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d, key=f"filt_{col}")
                if isinstance(dr, tuple) and len(dr) == 2:
                    start, end = dr
                    filtered = filtered[
                        (filtered[col] >= pd.Timestamp(start)) &
                        (filtered[col] < pd.Timestamp(end) + pd.Timedelta(days=1))
                    ]
            elif col in cols["numeric"]:
                s = df[col].dropna()
                if s.empty:
                    continue
                mn, mx = float(s.min()), float(s.max())
                if mn == mx:
                    st.caption(f"ستون «{col}» فقط یک مقدار دارد ({fa_num(mn)}) و قابل برش نیست.")
                    continue
                rng = st.slider(f"بازهٔ «{col}»", min_value=mn, max_value=mx, value=(mn, mx), key=f"filt_{col}")
                filtered = filtered[(filtered[col] >= rng[0]) & (filtered[col] <= rng[1])]
            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                selected = st.multiselect(f"مقادیر «{col}»", options, default=options, key=f"filt_{col}")
                filtered = filtered[filtered[col].astype(str).isin(selected)]

        st.caption(f"📌 پس از اعمال فیلتر: {fa_num(len(filtered))} ردیف از {fa_num(len(df))} ردیف باقی ماند.")
        return filtered


# ============================================================================
# نمودارهای پیشرفتهٔ اضافه‌شده (وتری، رادار مقایسه‌ای، حبابی، کارت، درختی، گیج و ...)
# ============================================================================
def make_chord_diagram(labels, matrix, colorway=PLOTLY_COLORWAY):
    """رسم نمودار وتری (Chord Diagram) صرفاً با Plotly (بدون کتابخانهٔ اضافه)."""
    n = len(labels)
    matrix = np.array(matrix, dtype=float)
    total_per_node = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    grand_total = total_per_node.sum()
    if grand_total <= 0 or n < 2:
        return None

    gap = (0.015 * 2 * np.pi) if n > 1 else 0
    node_angles = []
    start = 0.0
    for i in range(n):
        span = (total_per_node[i] / grand_total) * (2 * np.pi - n * gap)
        span = max(span, 1e-6)
        node_angles.append((start, start + span))
        start += span + gap

    def point(angle, r=1.0):
        return r * np.cos(angle), r * np.sin(angle)

    def bezier(p0, p1, p2, t):
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

    fig = go.Figure()
    for i, (a0, a1) in enumerate(node_angles):
        theta = np.linspace(a0, a1, 30)
        fig.add_trace(go.Scatter(
            x=1.03 * np.cos(theta), y=1.03 * np.sin(theta), mode="lines",
            line=dict(color=colorway[i % len(colorway)], width=12),
            hoverinfo="text", text=f"{labels[i]} (مجموع: {fa_num(total_per_node[i], 1)})",
            showlegend=False,
        ))
        mid = (a0 + a1) / 2
        lx, ly = point(mid, 1.14)
        fig.add_annotation(x=lx, y=ly, text=str(labels[i]), showarrow=False,
                            font=dict(size=12, color=TEXT_COLOR), xanchor="center")

    cursor = {i: node_angles[i][0] for i in range(n)}
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val <= 0:
                continue
            span_i = (val / total_per_node[i]) * (node_angles[i][1] - node_angles[i][0]) if total_per_node[i] > 0 else 0
            span_j = (val / total_per_node[j]) * (node_angles[j][1] - node_angles[j][0]) if total_per_node[j] > 0 else 0
            a_start, a_end = cursor[i], cursor[i] + span_i
            b_start, b_end = cursor[j], cursor[j] + span_j
            cursor[i] += span_i
            if i != j:
                cursor[j] += span_j

            t = np.linspace(0, 1, 20)
            p_a0, p_a1 = np.array(point(a_start)), np.array(point(a_end))
            p_b0, p_b1 = np.array(point(b_start)), np.array(point(b_end))
            center = np.array([0.0, 0.0])
            curve1 = np.array([bezier(p_a1, center, p_b0, tt) for tt in t])
            curve2 = np.array([bezier(p_b1, center, p_a0, tt) for tt in t])
            arc_a = np.array([point(ang) for ang in np.linspace(a_start, a_end, 10)])
            arc_b = np.array([point(ang) for ang in np.linspace(b_start, b_end, 10)])
            poly = np.vstack([arc_a, curve1, arc_b, curve2])

            fig.add_trace(go.Scatter(
                x=poly[:, 0], y=poly[:, 1], mode="lines", fill="toself",
                line=dict(width=0.5, color=colorway[i % len(colorway)]),
                fillcolor=colorway[i % len(colorway)], opacity=0.42,
                hoverinfo="text", text=f"{labels[i]} → {labels[j]}: {fa_num(val, 1)}",
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.45, 1.45]),
        yaxis=dict(visible=False, range=[-1.45, 1.45], scaleanchor="x", scaleratio=1),
        height=560, plot_bgcolor=CARD_COLOR, paper_bgcolor=CARD_COLOR,
    )
    return fig


def make_radar_comparison(df, metric_cols, subject_col=None, subjects_selected=None, agg="mean"):
    """رادار مقایسه‌ای؛ اگر subject_col داده شود، دو (یا چند) موضوع را روی هم مقایسه می‌کند."""
    fig = go.Figure()
    theta = metric_cols + [metric_cols[0]]

    if subject_col and subjects_selected:
        for subj in subjects_selected:
            sub_df = df[df[subject_col].astype(str) == str(subj)]
            values = []
            for m in metric_cols:
                v = sub_df[m].count() if agg == "count" else sub_df[m].agg(agg)
                values.append(float(v) if pd.notna(v) else 0.0)
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=str(subj)))
    else:
        values = []
        for m in metric_cols:
            v = df[m].count() if agg == "count" else df[m].agg(agg)
            values.append(float(v) if pd.notna(v) else 0.0)
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name="کل داده"))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=520,
                       showlegend=True)
    return fig


def make_info_cards_html(df, metric_cols, agg_label):
    agg_map = {"مجموع": "sum", "میانگین": "mean", "حداکثر": "max", "حداقل": "min", "تعداد": "count"}
    cards = []
    for col in metric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        val = s.count() if agg_label == "تعداد" else getattr(s, agg_map[agg_label])()
        cards.append(f"""
        <div style="flex:1;min-width:170px;background:{CARD_COLOR};border:1px solid {GRID_COLOR};
                    border-radius:12px;padding:16px 18px;box-shadow:0 1px 3px rgba(15,23,42,.06);">
          <div style="font-size:13px;color:#64748b;margin-bottom:6px;">{agg_label} {col}</div>
          <div style="font-size:26px;font-weight:800;color:{ACCENT_COLOR};">{fa_num(val, 2)}</div>
        </div>""")
    if not cards:
        return None
    return f'<div style="display:flex;gap:14px;flex-wrap:wrap;direction:rtl;">{"".join(cards)}</div>'


def make_pivot_heatmap(df, row_col, col_col, value_col, agg_func):
    if value_col == "تعداد (شمارش خودکار)":
        pivot = df.pivot_table(index=row_col, columns=col_col, values=df.columns[0], aggfunc="count")
    else:
        pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc=agg_func)
    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Blues", aspect="auto")
    return fig


def make_gauge_needle(value, min_v, max_v, title):
    mid = (min_v + max_v) / 2
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 18}},
        delta={"reference": mid},
        gauge={
            "axis": {"range": [min_v, max_v]},
            "bar": {"color": ACCENT_COLOR},
            "bgcolor": CARD_COLOR,
            "steps": [
                {"range": [min_v, min_v + (max_v - min_v) * 0.5], "color": "#e2e8f0"},
                {"range": [min_v + (max_v - min_v) * 0.5, min_v + (max_v - min_v) * 0.8], "color": "#cbd5e1"},
            ],
            "threshold": {"line": {"color": ACCENT_2, "width": 4}, "thickness": 0.85, "value": value},
        },
    ))
    fig.update_layout(height=380, paper_bgcolor=CARD_COLOR, font=dict(family=FONT_FAMILY, color=TEXT_COLOR))
    return fig


def make_half_gauge(value, min_v, max_v, title):
    value = max(min(value, max_v), min_v)
    span = max_v - min_v
    if span <= 0:
        return None
    filled = value - min_v
    remaining = span - filled
    fig = go.Figure(go.Pie(
        values=[filled, remaining, span],
        hole=0.65, rotation=90, direction="clockwise", sort=False,
        marker=dict(colors=[ACCENT_COLOR, GRID_COLOR, "rgba(0,0,0,0)"]),
        textinfo="none", showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{fa_num(value, 1)}</b><br><span style='font-size:12px;color:#64748b'>{title}</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=TEXT_COLOR),
        )],
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor=CARD_COLOR,
    )
    return fig


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

df = render_filter_panel(df, cols)
if df.empty:
    st.warning("با فیلترهای انتخاب‌شده هیچ داده‌ای باقی نماند. لطفاً فیلترها را تغییر بده.")
    st.stop()

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
        ["خطی (Line)", "میله‌ای (Bar)", "میله‌ای پشته‌ای (Stacked Bar)", "پراکنش (Scatter)",
         "پراکنش حبابی (Bubble)", "هیستوگرام", "باکس‌پلات (Box)", "ویولن (Violin)",
         "دایره‌ای (Pie)", "نقشه حرارتی همبستگی (Heatmap)", "نقشه حرارتی محوری (Heatmap Pivot)",
         "وتری (Chord Diagram)", "رادار مقایسه‌ای (Radar Compare)", "درختی (Treemap)",
         "کارت اطلاعاتی (Info Cards)", "گیج عقربه‌دار (Gauge)", "گیج نیم‌دایره‌ای (Half Gauge)"],
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

    elif chart_type == "میله‌ای پشته‌ای (Stacked Bar)":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("محور X (دسته‌بندی)", all_cols, key="stack_x")
        with c2:
            stack_y_cols = st.multiselect("ستون‌های عددی برای پشته‌سازی",
                                           numeric_cols_ext, default=numeric_cols_ext[:2], key="stack_y")

    elif chart_type == "پراکنش حبابی (Bubble)":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_col = st.selectbox("محور X", numeric_cols_ext, key="bub_x")
        with c2:
            y_options = [c for c in numeric_cols_ext if c != x_col]
            y_col = st.selectbox("محور Y", y_options, key="bub_y")
        with c3:
            size_options = [c for c in numeric_cols_ext if c not in (x_col, y_col)]
            size_col = st.selectbox("اندازهٔ حباب بر اساس", size_options, key="bub_size") if size_options else None
        with c4:
            color_options = ["— هیچکدام —"] + [c for c in all_cols if c not in (x_col, y_col)]
            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)", color_options, key="bub_color")
            color_col = None if color_col == "— هیچکدام —" else color_col

    elif chart_type == "نقشه حرارتی محوری (Heatmap Pivot)":
        c1, c2, c3 = st.columns(3)
        with c1:
            row_col = st.selectbox("ستون سطرها", cols["categorical"] + cols["datetime"], key="pv_row")
        with c2:
            col_options = [c for c in (cols["categorical"] + cols["datetime"]) if c != row_col]
            col_col = st.selectbox("ستون ستون‌ها", col_options, key="pv_col")
        with c3:
            value_options = ["تعداد (شمارش خودکار)"] + numeric_cols_ext
            value_col = st.selectbox("ستون مقدار", value_options, key="pv_val")
        pivot_agg = st.selectbox("نحوهٔ تجمیع", ["mean", "sum", "median", "max", "min"], key="pv_agg")

    elif chart_type == "وتری (Chord Diagram)":
        c1, c2, c3 = st.columns(3)
        with c1:
            chord_source = st.selectbox("ستون مبدأ", cols["categorical"], key="chord_src")
        with c2:
            chord_target_options = [c for c in cols["categorical"] if c != chord_source]
            chord_target = st.selectbox("ستون مقصد", chord_target_options, key="chord_tgt") if chord_target_options else None
        with c3:
            chord_value_options = ["تعداد (شمارش خودکار)"] + numeric_cols_ext
            chord_value = st.selectbox("ستون مقدار (وزن رابطه)", chord_value_options, key="chord_val")
        chord_top_n = st.slider("حداکثر تعداد گره‌های نمایش‌داده‌شده (برای خوانایی)", 3, 20, 10, key="chord_topn")

    elif chart_type == "رادار مقایسه‌ای (Radar Compare)":
        radar_metrics = st.multiselect("ستون‌های عددی (محورهای رادار)", numeric_cols_ext,
                                        default=numeric_cols_ext[:5], key="radar_metrics")
        c1, c2 = st.columns(2)
        with c1:
            radar_subject_col = st.selectbox("ستون موضوع برای مقایسه (اختیاری)",
                                              ["— بدون مقایسه —"] + cols["categorical"], key="radar_subject")
            radar_subject_col = None if radar_subject_col == "— بدون مقایسه —" else radar_subject_col
        with c2:
            radar_agg = st.selectbox("نحوهٔ تجمیع مقادیر", ["mean", "sum", "median", "max", "min", "count"], key="radar_agg")
        radar_subjects = []
        if radar_subject_col:
            subj_options = sorted(df[radar_subject_col].dropna().astype(str).unique().tolist())
            radar_subjects = st.multiselect("موضوع‌هایی که می‌خواهی مقایسه کنی (حداقل ۱، پیشنهاد: ۲)",
                                             subj_options, default=subj_options[:2], key="radar_subjects")

    elif chart_type == "درختی (Treemap)":
        tree_path_cols = st.multiselect("ستون‌های سلسله‌مراتبی (از کلی به جزئی)",
                                         cols["categorical"], default=cols["categorical"][:2], key="tree_path")
        tree_value_options = ["تعداد (شمارش خودکار)"] + numeric_cols_ext
        tree_value_col = st.selectbox("ستون مقدار", tree_value_options, key="tree_val")

    elif chart_type == "کارت اطلاعاتی (Info Cards)":
        card_metrics = st.multiselect("ستون‌های عددی برای نمایش در کارت", numeric_cols_ext,
                                       default=numeric_cols_ext[:4], key="card_metrics")
        card_agg = st.selectbox("نوع شاخص نمایش‌داده‌شده", ["مجموع", "میانگین", "حداکثر", "حداقل", "تعداد"], key="card_agg")

    elif chart_type in ["گیج عقربه‌دار (Gauge)", "گیج نیم‌دایره‌ای (Half Gauge)"]:
        c1, c2 = st.columns(2)
        with c1:
            gauge_metric = st.selectbox("ستون عددی", numeric_cols_ext, key="gauge_metric")
            gauge_agg = st.selectbox("نحوهٔ محاسبهٔ مقدار", ["mean", "sum", "median", "max", "min"], key="gauge_agg")
        with c2:
            gauge_min = st.number_input("حداقل بازه گیج", value=0.0, key="gauge_min")
            gauge_max_default = float(df[gauge_metric].dropna().max()) if gauge_metric and not df[gauge_metric].dropna().empty else 100.0
            gauge_max = st.number_input("حداکثر بازه گیج", value=gauge_max_default, key="gauge_max")

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

            elif chart_type == "میله‌ای پشته‌ای (Stacked Bar)":
                if not stack_y_cols:
                    st.warning("حداقل یک ستون عددی برای پشته‌سازی انتخاب کنید.")
                else:
                    melted = df.melt(id_vars=[x_col], value_vars=stack_y_cols,
                                      var_name="سری", value_name="مقدار")
                    fig = px.bar(melted, x=x_col, y="مقدار", color="سری", barmode="stack")
                    if len(stack_y_cols) == 1:
                        analysis_type, analysis_params = "anova", {"num_col": stack_y_cols[0], "cat_col": x_col}

            elif chart_type == "پراکنش حبابی (Bubble)":
                if not size_col:
                    st.warning("برای نمودار حبابی حداقل به سه ستون عددی (X، Y، اندازه) نیاز است.")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                                      size_max=45, hover_name=color_col)
                    analysis_type, analysis_params = "regression", {"x_col": x_col, "y_col": y_col}

            elif chart_type == "نقشه حرارتی محوری (Heatmap Pivot)":
                if not col_col:
                    st.warning("ستون‌های سطر و ستون باید متفاوت باشند.")
                else:
                    fig = make_pivot_heatmap(df, row_col, col_col, value_col, pivot_agg)

            elif chart_type == "وتری (Chord Diagram)":
                if not chord_target:
                    st.warning("ستون‌های مبدأ و مقصد باید متفاوت باشند.")
                else:
                    if chord_value == "تعداد (شمارش خودکار)":
                        flow = df.groupby([chord_source, chord_target]).size().reset_index(name="_val")
                    else:
                        flow = df.groupby([chord_source, chord_target])[chord_value].sum().reset_index(name="_val")
                    top_labels = pd.concat([flow[chord_source], flow[chord_target]]).value_counts().head(chord_top_n).index.tolist()
                    flow = flow[flow[chord_source].isin(top_labels) & flow[chord_target].isin(top_labels)]
                    labels = sorted(set(flow[chord_source]) | set(flow[chord_target]), key=str)
                    idx = {lab: i for i, lab in enumerate(labels)}
                    matrix = np.zeros((len(labels), len(labels)))
                    for _, row in flow.iterrows():
                        matrix[idx[row[chord_source]], idx[row[chord_target]]] += row["_val"]
                    fig = make_chord_diagram(labels, matrix)
                    if fig is None:
                        st.warning("داده کافی برای رسم نمودار وتری وجود ندارد.")

            elif chart_type == "رادار مقایسه‌ای (Radar Compare)":
                if len(radar_metrics) < 3:
                    st.warning("برای رادار حداقل به سه محور (ستون عددی) نیاز است.")
                else:
                    fig = make_radar_comparison(df, radar_metrics, radar_subject_col, radar_subjects, radar_agg)

            elif chart_type == "درختی (Treemap)":
                if not tree_path_cols:
                    st.warning("حداقل یک ستون برای مسیر سلسله‌مراتبی انتخاب کنید.")
                else:
                    if tree_value_col == "تعداد (شمارش خودکار)":
                        fig = px.treemap(df, path=[px.Constant("همه")] + tree_path_cols)
                    else:
                        fig = px.treemap(df, path=[px.Constant("همه")] + tree_path_cols, values=tree_value_col)
                    fig.update_traces(root_color=CARD_COLOR)

            elif chart_type == "کارت اطلاعاتی (Info Cards)":
                if not card_metrics:
                    st.warning("حداقل یک ستون عددی انتخاب کنید.")
                else:
                    fig = make_info_cards_html(df, card_metrics, card_agg)
                    if fig is None:
                        st.warning("داده کافی برای ساخت کارت وجود ندارد.")

            elif chart_type in ["گیج عقربه‌دار (Gauge)", "گیج نیم‌دایره‌ای (Half Gauge)"]:
                s = df[gauge_metric].dropna()
                if s.empty:
                    st.warning("داده کافی برای این ستون وجود ندارد.")
                else:
                    val = float(s.count()) if gauge_agg == "count" else float(s.agg(gauge_agg))
                    label = plot_title or gauge_metric
                    if chart_type == "گیج عقربه‌دار (Gauge)":
                        fig = make_gauge_needle(val, gauge_min, gauge_max, label)
                    else:
                        fig = make_half_gauge(val, gauge_min, gauge_max, label)

            if fig is not None:
                is_html_content = chart_type == "کارت اطلاعاتی (Info Cards)"
                if plot_title and not is_html_content:
                    fig.update_layout(title=plot_title)
                st.session_state["_manual_fig"] = fig
                st.session_state["_manual_analysis_type"] = analysis_type
                st.session_state["_manual_analysis_params"] = analysis_params
                st.session_state["_manual_title"] = plot_title or chart_type
                st.session_state["_manual_is_html"] = is_html_content
        except Exception as e:
            st.error(f"خطا در رسم نمودار: {e}")

    if st.session_state.get("_manual_fig") is not None:
        atype = st.session_state.get("_manual_analysis_type")
        aparams = st.session_state.get("_manual_analysis_params") or {}
        is_html_content = st.session_state.get("_manual_is_html", False)
        if atype:
            analysis = run_analysis({"type": atype, "params": aparams}, df)
        else:
            analysis = {"error": "برای این ترکیب نمودار، تحلیل آماری خودکار در دسترس نیست."}
        render_analysis_block("manual_current", st.session_state.get("_manual_title", "نمودار دلخواه"),
                               st.session_state["_manual_fig"], atype or "none", analysis, is_html=is_html_content)

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
