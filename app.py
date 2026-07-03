# -*- coding: utf-8 -*-
"""
Smart Chart Explorer
---------------------
یک اپلیکیشن استریم‌لیت که با گرفتن هر فایل داده‌ای (CSV / Excel):
  1) ساختار داده (نوع ستون‌ها، تعداد یکتا، وجود تاریخ و ...) را تحلیل می‌کند
  2) به‌صورت خودکار مناسب‌ترین نمودار(های) توصیفی را پیشنهاد و رسم می‌کند
  3) امکان ساخت نمودار دلخواه کاربر (انتخاب نوع نمودار، محورها، رنگ، تجمیع و ...) را فراهم می‌کند

اجرا:
    pip install streamlit pandas numpy plotly openpyxl
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# تنظیمات کلی صفحه
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Chart Explorer", layout="wide")
st.title("📊 Smart Chart Explorer")
st.caption("داده‌ات رو بده، ما بهترین نمودار توصیفی رو پیدا می‌کنیم — یا خودت انتخاب کن.")


# ----------------------------------------------------------------------------
# توابع کمکی: بارگذاری و تحلیل ساختار داده
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """خواندن فایل CSV یا Excel و تلاش برای تشخیص خودکار نوع ستون‌های تاریخ."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("فرمت فایل پشتیبانی نمی‌شود. لطفاً CSV یا Excel آپلود کنید.")

    # تلاش برای تبدیل ستون‌های متنی که شبیه تاریخ هستند
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
    """دسته‌بندی ستون‌ها به عددی، دسته‌ای، تاریخی و بولی."""
    numeric_cols, categorical_cols, datetime_cols, boolean_cols = [], [], [], []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            # اگر تعداد مقادیر یکتا خیلی کم باشد، می‌تواند دسته‌ای رفتار کند (مثل کد استان)
            if s.nunique(dropna=True) <= 10 and s.dropna().shape[0] > 0:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
    }


# ----------------------------------------------------------------------------
# موتور تشخیص هوشمند نمودار
# ----------------------------------------------------------------------------
def suggest_charts(df: pd.DataFrame, cols: dict) -> list:
    """
    بر اساس ساختار داده، فهرستی از نمودارهای پیشنهادی می‌سازد.
    هر آیتم: (عنوان, تابع سازنده نمودار)
    """
    suggestions = []
    numeric = cols["numeric"]
    categorical = cols["categorical"]
    datetime_cols = cols["datetime"]
    n_rows = len(df)

    # 1) سری زمانی: تاریخ + عددی -> نمودار خطی
    if datetime_cols and numeric:
        dcol = datetime_cols[0]
        for ncol in numeric[:2]:
            df_sorted = df.sort_values(dcol)
            fig = px.line(df_sorted, x=dcol, y=ncol, markers=True,
                          title=f"روند {ncol} در طول زمان ({dcol})")
            suggestions.append((f"روند زمانی: {ncol} بر حسب {dcol}", fig))

    # 2) یک ستون عددی به تنهایی -> هیستوگرام (توزیع)
    if numeric:
        ncol = numeric[0]
        fig = px.histogram(df, x=ncol, nbins=30, marginal="box",
                            title=f"توزیع {ncol}")
        suggestions.append((f"توزیع (هیستوگرام) {ncol}", fig))

    # 3) دو ستون عددی -> پراکنش (اسکترپلات) + خط روند
    if len(numeric) >= 2:
        x, y = numeric[0], numeric[1]
        color = categorical[0] if categorical else None
        fig = px.scatter(df, x=x, y=y, color=color, trendline="ols" if n_rows < 5000 else None,
                          title=f"رابطهٔ {x} و {y}")
        suggestions.append((f"پراکنش: {x} در برابر {y}", fig))

    # 4) یک ستون دسته‌ای -> نمودار میله‌ای فراوانی
    if categorical:
        ccol = categorical[0]
        vc = df[ccol].value_counts(dropna=False).reset_index()
        vc.columns = [ccol, "تعداد"]
        vc = vc.head(20)
        fig = px.bar(vc, x=ccol, y="تعداد", title=f"فراوانی مقادیر {ccol}")
        suggestions.append((f"فراوانی دسته‌ها: {ccol}", fig))

        # اگر تعداد دسته‌ها کم باشد، نمودار دایره‌ای هم مفید است
        if df[ccol].nunique(dropna=True) <= 8:
            fig_pie = px.pie(vc, names=ccol, values="تعداد", title=f"سهم هر دسته در {ccol}")
            suggestions.append((f"نمودار دایره‌ای: {ccol}", fig_pie))

    # 5) یک ستون عددی + یک دسته‌ای -> باکس‌پلات مقایسه‌ای
    if numeric and categorical:
        ncol, ccol = numeric[0], categorical[0]
        if df[ccol].nunique(dropna=True) <= 15:
            fig = px.box(df, x=ccol, y=ncol, color=ccol,
                         title=f"مقایسهٔ توزیع {ncol} بر اساس {ccol}")
            suggestions.append((f"مقایسه دسته‌ای: {ncol} بر اساس {ccol}", fig))

    # 6) چند ستون عددی -> ماتریس همبستگی (Heatmap)
    if len(numeric) >= 3:
        corr = df[numeric].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="ماتریس همبستگی متغیرهای عددی")
        suggestions.append(("ماتریس همبستگی", fig))

    # اگر هیچ پیشنهادی ساخته نشد، حداقل نمای کلی جدول را نشان بده
    if not suggestions:
        suggestions.append(("پیش‌نمایش داده", px.scatter(title="داده کافی برای تشخیص نمودار یافت نشد")))

    return suggestions


# ----------------------------------------------------------------------------
# رابط کاربری اصلی
# ----------------------------------------------------------------------------
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

with st.expander("🔍 پیش‌نمایش و ساختار داده", expanded=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(50), use_container_width=True)
    with c2:
        st.write(f"**تعداد ردیف‌ها:** {len(df):,}")
        st.write(f"**تعداد ستون‌ها:** {df.shape[1]}")
        st.write(f"**ستون‌های عددی:** {', '.join(cols['numeric']) or '—'}")
        st.write(f"**ستون‌های دسته‌ای:** {', '.join(cols['categorical']) or '—'}")
        st.write(f"**ستون‌های تاریخی:** {', '.join(cols['datetime']) or '—'}")
        st.write(f"**ستون‌های بولی:** {', '.join(cols['boolean']) or '—'}")

tab_auto, tab_manual = st.tabs(["🤖 نمودار پیشنهادی خودکار", "🛠️ نمودار دلخواه من"])

# --------------------------- تب ۱: تشخیص خودکار -----------------------------
with tab_auto:
    st.subheader("نمودارهای پیشنهادشده بر اساس ساختار داده")
    suggestions = suggest_charts(df, cols)
    st.caption(f"{len(suggestions)} نمودار بر اساس تحلیل خودکار داده پیشنهاد شد.")

    for i in range(0, len(suggestions), 2):
        row = suggestions[i:i + 2]
        chart_cols = st.columns(len(row))
        for col_ui, (title, fig) in zip(chart_cols, row):
            with col_ui:
                st.markdown(f"**{title}**")
                st.plotly_chart(fig, use_container_width=True)

# --------------------------- تب ۲: نمودار دلخواه -----------------------------
with tab_manual:
    st.subheader("نمودار خودت رو بساز")

    chart_type = st.selectbox(
        "نوع نمودار",
        ["خطی (Line)", "میله‌ای (Bar)", "پراکنش (Scatter)", "هیستوگرام",
         "باکس‌پلات (Box)", "ویولن (Violin)", "دایره‌ای (Pie)", "نقشه حرارتی همبستگی (Heatmap)"],
    )

    all_cols = list(df.columns)
    numeric_cols = cols["numeric"] + cols["boolean"]
    all_for_axis = all_cols

    x_col = y_col = color_col = size_col = None
    agg_func = None

    if chart_type in ["خطی (Line)", "میله‌ای (Bar)", "پراکنش (Scatter)", "باکس‌پلات (Box)", "ویولن (Violin)"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("محور X", all_for_axis, key="x")
        with c2:
            y_options = [c for c in all_for_axis if c != x_col]
            y_col = st.selectbox("محور Y", y_options, key="y")
        with c3:
            color_options = ["— هیچکدام —"] + [c for c in all_for_axis if c not in (x_col, y_col)]
            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)", color_options, key="color")
            color_col = None if color_col == "— هیچکدام —" else color_col

        if chart_type in ["میله‌ای (Bar)", "خطی (Line)"]:
            agg_func = st.selectbox("نحوهٔ تجمیع مقادیر Y در صورت تکرار X",
                                     ["بدون تجمیع (مقادیر خام)", "میانگین", "مجموع", "تعداد", "میانه"])

    elif chart_type == "هیستوگرام":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("ستون برای بررسی توزیع", all_for_axis, key="hist_x")
        with c2:
            color_options = ["— هیچکدام —"] + [c for c in all_for_axis if c != x_col]
            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)", color_options, key="hist_color")
            color_col = None if color_col == "— هیچکدام —" else color_col
        bins = st.slider("تعداد بازه‌ها (bins)", 5, 100, 30)

    elif chart_type == "دایره‌ای (Pie)":
        c1, c2 = st.columns(2)
        with c1:
            names_col = st.selectbox("ستون دسته‌بندی (برچسب‌ها)", all_for_axis, key="pie_names")
        with c2:
            value_options = ["تعداد (شمارش خودکار)"] + numeric_cols
            values_col = st.selectbox("ستون مقدار", value_options, key="pie_values")

    elif chart_type == "نقشه حرارتی همبستگی (Heatmap)":
        selected_numeric = st.multiselect("ستون‌های عددی برای محاسبه همبستگی",
                                           numeric_cols, default=numeric_cols[:6])

    plot_title = st.text_input("عنوان نمودار (اختیاری)", value="")

    if st.button("رسم نمودار", type="primary"):
        try:
            fig = None

            if chart_type in ["خطی (Line)", "میله‌ای (Bar)"] and agg_func and agg_func != "بدون تجمیع (مقادیر خام)":
                agg_map = {"میانگین": "mean", "مجموع": "sum", "تعداد": "count", "میانه": "median"}
                group_cols = [x_col] + ([color_col] if color_col else [])
                plot_df = df.groupby(group_cols, dropna=False)[y_col].agg(agg_map[agg_func]).reset_index()
            else:
                plot_df = df

            if chart_type == "خطی (Line)":
                fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, markers=True)
            elif chart_type == "میله‌ای (Bar)":
                fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, barmode="group")
            elif chart_type == "پراکنش (Scatter)":
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col)
            elif chart_type == "هیستوگرام":
                fig = px.histogram(plot_df, x=x_col, color=color_col, nbins=bins)
            elif chart_type == "باکس‌پلات (Box)":
                fig = px.box(plot_df, x=x_col, y=y_col, color=color_col)
            elif chart_type == "ویولن (Violin)":
                fig = px.violin(plot_df, x=x_col, y=y_col, color=color_col, box=True)
            elif chart_type == "دایره‌ای (Pie)":
                if values_col == "تعداد (شمارش خودکار)":
                    vc = df[names_col].value_counts(dropna=False).reset_index()
                    vc.columns = [names_col, "تعداد"]
                    fig = px.pie(vc, names=names_col, values="تعداد")
                else:
                    fig = px.pie(df, names=names_col, values=values_col)
            elif chart_type == "نقشه حرارتی همبستگی (Heatmap)":
                if len(selected_numeric) < 2:
                    st.warning("حداقل دو ستون عددی برای محاسبه همبستگی انتخاب کنید.")
                else:
                    corr = df[selected_numeric].corr(numeric_only=True)
                    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)

            if fig is not None:
                if plot_title:
                    fig.update_layout(title=plot_title)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"خطا در رسم نمودار: {e}")

st.divider()
st.caption("ساخته‌شده با Streamlit · Pandas · NumPy · Plotly")
