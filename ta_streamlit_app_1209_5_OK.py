import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# ---------- 常態分佈相關函數 ----------

def normal_cdf(z: float) -> float:
    """標準常態分佈 CDF Φ(z)"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def central_yield_by_cpk(cpk: float) -> float:
    """依 Cpk 計算中心良率"""
    if cpk is None or pd.isna(cpk):
        return np.nan
    z = 3.0 * cpk
    return normal_cdf(z) - normal_cdf(-z)


def classify_cpk(cpk: float) -> str:
    """Cpk 等級判斷（只用在畫面顯示，不進 PDF）"""
    if cpk is None or pd.isna(cpk):
        return ""
    if cpk >= 1.67:
        return "完美(Perfect)"
    if cpk < 1.0:
        return "高風險(High Risk)"
    if cpk >= 1.33:
        return "理想(Ideal)"
    return "可接受(Acceptable)"


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        return 0.0
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def index_to_label(idx: int) -> str:
    """0→A, 1→B… 26→AA"""
    letters = []
    n = idx
    while True:
        n, r = divmod(n, 26)
        letters.append(chr(ord("A") + r))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(letters))


# ---------- 欄位名稱 ----------

COL_PATH = "公差路徑 (Tolerance Loop)"
COL_DIM = "Dimension (mm)"       # 設計尺寸
COL_TOL = "公差(±T)"
COL_CPK = "CPK"
COL_LABEL = "公差路徑代號"
COL_DELETE = "刪除"


# ---------- Streamlit App ----------

st.set_page_config(
    page_title="TA Template (Streamlit 版)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 公差分析 TA Template – Sercomm 版")

# ---- 基本資訊 ----

st.subheader("基本資料填寫")

# Project & Engineer（同一行）
c1, c2 = st.columns(2)
with c1:
    project_name = st.text_input("Project", value="", placeholder="輸入專案名稱")
with c2:
    engineer_name = st.text_input("Engineer", value="", placeholder="輸入工程師姓名")

# Title
title = st.text_input("Title", value="", placeholder="輸入計算主題")

# TA Loop 圖片
st.markdown("**TA Loop 圖示（可拖拉圖片到此處上傳）**")
ta_loop_image = st.file_uploader(
    "將 TA Loop 截圖拖拉到這裡，或點擊選擇檔案",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)
if ta_loop_image:
    st.markdown("### TA Loop 圖示預覽")
    st.image(ta_loop_image, width=600)

st.markdown("---")


# ---- 1. 零件公差 + Cpk ----

st.subheader("1️⃣ 零件公差與 Cpk 輸入")
st.caption("請在下表輸入各零件的設計尺寸、公差 ±T (mm) 與 Cpk，代號會自動產生。")

# 初始化：一開始 base_df 為空，按「新增一列」才會出現 A 列
if "base_df" not in st.session_state:
    st.session_state["base_df"] = pd.DataFrame(
        columns=[COL_LABEL, COL_PATH, COL_DIM, COL_TOL, COL_CPK]
    )

base_df = st.session_state["base_df"]

toolbar_placeholder = st.empty()
with toolbar_placeholder.container():
    tcol1, tcol2, _ = st.columns([1, 1, 8])
    with tcol1:
        add_clicked = st.button("➕ 新增一列", key="btn_add_row", use_container_width=True)
    with tcol2:
        del_clicked = st.button("🗑 刪除勾選列", key="btn_delete_rows", use_container_width=True)

# 顯示表格或提示
if base_df.empty:
    st.info("請先按「➕ 新增一列」開始建立第一筆公差資料。")
    edited_df = base_df.copy()
else:
    editor_df = base_df.copy()
    if COL_DELETE not in editor_df.columns:
        editor_df[COL_DELETE] = False

    # 依列數動態調整表格高度，避免一大片空白
    row_height = 32
    base_height = 60
    table_rows = max(len(editor_df), 1)
    dynamic_height = base_height + row_height * table_rows

    edited_df = st.data_editor(
        editor_df,
        num_rows="fixed",
        use_container_width=True,
        key="ta_input_editor",
        hide_index=True,
        height=dynamic_height,
        column_config={
            COL_LABEL: st.column_config.TextColumn(
                "公差路徑代號",
                disabled=True,
                width="small",
            ),
            COL_PATH: st.column_config.TextColumn(
                COL_PATH,
                help="請描述公差路徑",
            ),
            COL_DIM: st.column_config.NumberColumn(
                "Dimension (mm)",
                step=0.01,
                format="%.2f",
                help="請輸入設計尺寸",
            ),
            COL_TOL: st.column_config.NumberColumn(
                "公差 (±T)",
                step=0.001,
                format="%.3f",
                help="請設定公差值",
            ),
            COL_CPK: st.column_config.NumberColumn(
                "單件 Cpk",
                step=0.01,
                format="%.2f",
                help="請輸入單件目標 Cpk",
            ),
            COL_DELETE: st.column_config.CheckboxColumn(
                "刪除",
                width="small",
            ),
        },
    )

# ➕ 新增一列
if add_clicked:
    if base_df.empty:
        # 從完全空白建第一列 A
        df_tmp = pd.DataFrame(
            [{
                COL_LABEL: index_to_label(0),
                COL_PATH: "",
                COL_DIM: np.nan,
                COL_TOL: np.nan,
                COL_CPK: np.nan,
            }]
        )
    else:
        # 以目前畫面上的內容為基準新增
        df_tmp = edited_df.copy()
        if COL_DELETE in df_tmp.columns:
            df_tmp = df_tmp.drop(columns=[COL_DELETE])
        df_tmp = df_tmp.reset_index(drop=True)

        next_label = index_to_label(len(df_tmp))
        new_row = {
            COL_LABEL: next_label,
            COL_PATH: "",
            COL_DIM: np.nan,
            COL_TOL: np.nan,
            COL_CPK: np.nan,
        }
        df_tmp = pd.concat([df_tmp, pd.DataFrame([new_row])], ignore_index=True)

    st.session_state["base_df"] = df_tmp
    st.rerun()

# 🗑 刪除勾選列
if del_clicked and not base_df.empty:
    df_new = edited_df.copy()
    if COL_DELETE in df_new.columns:
        df_new = df_new[~df_new[COL_DELETE]].drop(columns=[COL_DELETE])

    df_new = df_new.reset_index(drop=True)
    df_new[COL_LABEL] = [index_to_label(i) for i in range(len(df_new))]
    df_new = df_new[[COL_LABEL, COL_PATH, COL_DIM, COL_TOL, COL_CPK]]

    st.session_state["base_df"] = df_new
    st.rerun()

# 後續計算使用 edited_df
df_calc = edited_df.copy()
if COL_DELETE in df_calc.columns:
    df_calc = df_calc.drop(columns=[COL_DELETE])

df_calc[COL_DIM] = pd.to_numeric(df_calc[COL_DIM], errors="coerce")
df_calc[COL_TOL] = pd.to_numeric(df_calc[COL_TOL], errors="coerce")
df_calc[COL_CPK] = pd.to_numeric(df_calc[COL_CPK], errors="coerce")

mask_valid = ~(df_calc[COL_TOL].isna() & df_calc[COL_CPK].isna())
df_calc = df_calc[mask_valid].reset_index(drop=True)

if df_calc.empty:
    st.warning("請至少輸入一筆公差或 Cpk。")
    st.stop()


# ---------- 2. 單件結果 ----------

sigma_list, yield_list, ppm_list, remark_list = [], [], [], []

for _, row in df_calc.iterrows():
    T = row[COL_TOL]
    cpk = row[COL_CPK]

    if pd.isna(T) or pd.isna(cpk) or cpk == 0:
        sigma = y = ppm = np.nan
        remark = ""
    else:
        sigma = T / (3.0 * cpk)
        y = central_yield_by_cpk(cpk)
        ppm = (1.0 - y) * 1_000_000.0
        remark = classify_cpk(cpk)

    sigma_list.append(sigma)
    yield_list.append(y)
    ppm_list.append(ppm)
    remark_list.append(remark)

df_calc["σ"] = sigma_list
df_calc["理論良率(Yield)"] = yield_list
df_calc["不良率 (PPM)"] = ppm_list
df_calc["備註 Remark"] = remark_list
df_calc["理論良率(Yield %)"] = (
    df_calc["理論良率(Yield)"] * 100.0
).round(5).astype(str) + "%"

st.subheader("2️⃣ 各零件標準差與良率預估")
st.dataframe(
    df_calc[
        [
            COL_LABEL,
            COL_PATH,
            COL_DIM,
            COL_TOL,
            COL_CPK,
            "σ",
            "理論良率(Yield %)",
            "不良率 (PPM)",
            "備註 Remark",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# ---------- 3. RSS / Worst Case 累積公差 ----------

T_values = df_calc[COL_TOL].fillna(0.0).to_numpy()
sigma_values = df_calc["σ"].fillna(0.0).to_numpy()

tol_rss = float(np.sqrt(np.sum(T_values**2)))
tol_wc = float(np.sum(T_values))
sigma_stack = float(np.sqrt(np.sum(sigma_values**2)))

if sigma_stack > 0 and tol_rss > 0:
    cpk_stack_rss = tol_rss / (3.0 * sigma_stack)
    yield_stack_rss = central_yield_by_cpk(cpk_stack_rss)
else:
    cpk_stack_rss = yield_stack_rss = np.nan

st.subheader("3️⃣ 累積公差：Worst Case & RSS")

# ⭐ 顯示順序改為：Worst Case、RSS、Cpk、Yield
col1, col2, col3, col4 = st.columns(4)
col1.metric("Worst Case 累積公差", f"{tol_wc:.5f}")
col2.metric("RSS 累積公差", f"{tol_rss:.5f}")
col3.metric(
    "RSS Stack Cpk", f"{cpk_stack_rss:.3f}" if not math.isnan(cpk_stack_rss) else "-"
)
col4.metric(
    "RSS 預估良率",
    f"{yield_stack_rss*100:.5f}%" if not math.isnan(yield_stack_rss) else "-",
)


# ---------- 3-1. RSS 倍率試算 ----------

st.markdown("#### 📈 RSS 倍率對應公差與良率試算")

rss_factor_data = None  # 給 PDF 報告用

if not math.isnan(cpk_stack_rss) and sigma_stack > 0 and tol_rss > 0:
    rss_factor = st.number_input(
        "RSS 倍率",
        min_value=1.0,
        max_value=1.5001,   # 避免浮點誤差卡在 1.4
        step=0.1,
        value=1.0,
        format="%.1f",
    )

    rss_factor = round(rss_factor * 10) / 10.0
    rss_factor = max(1.0, min(1.5, rss_factor))

    rss_tol_scaled = tol_rss * rss_factor
    cpk_scaled = cpk_stack_rss * rss_factor
    yield_scaled = central_yield_by_cpk(cpk_scaled)

    df_rss = pd.DataFrame(
        {
            "x RSS": [f"{rss_factor:.1f}"],
            "Tol (mm)": [f"{rss_tol_scaled:.5f}"],
            "Yield Rate (%)": [f"{yield_scaled * 100.0:.5f}%"],
        }
    )
    st.dataframe(df_rss, hide_index=True)

    rss_factor_data = {
        "rss_factor": rss_factor,
        "rss_tol_scaled": rss_tol_scaled,
        "yield_scaled": yield_scaled,
    }


# ---------- 4. Sigma 對照表 ----------

st.subheader("4️⃣ 累積公差與 6σ 良率估算")

df_sigma = None  # 給 PDF 報告用

if sigma_stack > 0:
    rows_sigma = []
    for k in range(1, 7):
        tol_k = k * sigma_stack
        y = normal_cdf(k) - normal_cdf(-k)
        ppm = (1 - y) * 1_000_000.0

        if k <= 2:
            level = "不可接受\nUnacceptable"
            remark = ""
        elif k == 3:
            level = "最低標準\nMinimum Acceptable"
            remark = "Short Term"
        elif k == 4:
            level = "優良\nExcellent"
            remark = "Long Term"
        elif k == 5:
            level = "優良\nExcellent"
            remark = ""
        else:
            level = "完美\nPerfect"
            remark = ""

        rows_sigma.append(
            {
                "Sigma Level": f"± {k}σ",
                "Tol Stack": tol_k,
                "理論良率 (Yield %)": f"{y*100:.5f}%",
                "不良率 (PPM)": ppm,
                "良率級別": level,
                "備註": remark,
            }
        )

    df_sigma = pd.DataFrame(rows_sigma)
    st.dataframe(df_sigma, hide_index=True)


# ---------- 5. Normal Plot（畫面用：有灰線 + Sigma 標示） ----------

st.subheader("5️⃣ Normal Plot")

if sigma_stack > 0 and tol_wc != 0:
    k_max = min(6.0, abs(tol_wc) / sigma_stack)
else:
    k_max = 0.0

if k_max <= 0:
    st.info("Normal Plot 無法計算，請確認公差與 Cpk 是否合理。")
else:
    x_min = -k_max * sigma_stack
    x_max = k_max * sigma_stack

    num_ticks = 9
    x_ticks = np.linspace(x_min, x_max, num_ticks)

    x_values = np.linspace(x_min, x_max, 420)
    y_values = [normal_pdf(x, mu=0.0, sigma=sigma_stack) for x in x_values]
    y_max = max(y_values)

    fig, ax = plt.subplots(figsize=(5.304, 2.652))

    ax.plot(
        x_values,
        y_values,
        linewidth=0.8,
        color="#1a76d2",
        zorder=3,
    )

    label_y = 1.015
    for k in range(1, 7):
        for sign in (-1, 1):
            xk = sign * k * sigma_stack
            if x_min <= xk <= x_max:
                ax.axvline(
                    xk,
                    linestyle="--",
                    linewidth=0.5,
                    color="#bbbbbb",
                    zorder=2,
                )
                ax.text(
                    xk,
                    label_y,
                    f"{'+' if sign > 0 else '-'}{k}σ",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#444444",
                )

    ax.set_xlabel("Tolerance (mm)", fontsize=7)
    ax.set_ylabel("PDF", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f"{x:.3f}" for x in x_ticks],
        fontsize=7,
        rotation=0,
    )

    ax.set_ylim(bottom=0, top=y_max * 1.09)

    ax.grid(True, linestyle="--", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.25)
    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# ---------- 6. 產生 PDF 報告（Landscape + 簡潔 Normal Plot） ----------

st.subheader("6️⃣ 匯出 PDF 報告")
st.caption("將目前頁面上的 TA 資訊完整匯出為 PDF 報告（橫向）。")


def ascii_only(s: str) -> str:
    """只保留 ASCII 字元（避免中文在某些 PDF viewer 變黑框）"""
    return "".join(ch for ch in str(s) if ord(ch) < 128)


def generate_clean_normal_plot(sigma_stack_val, tol_wc_val):
    """產生 PDF 報告用的簡潔版 Normal Plot（無灰線、無 sigma 標線）"""
    if sigma_stack_val <= 0 or tol_wc_val == 0:
        return None

    k_max_local = min(6.0, abs(tol_wc_val) / sigma_stack_val)
    x_min_local = -k_max_local * sigma_stack_val
    x_max_local = k_max_local * sigma_stack_val

    x_vals = np.linspace(x_min_local, x_max_local, 500)
    y_vals = [normal_pdf(x, 0, sigma_stack_val) for x in x_vals]

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.0))

    ax2.plot(
        x_vals,
        y_vals,
        linewidth=1.0,
        color="#004C99"
    )

    ax2.set_xlabel("Tolerance (mm)", fontsize=8)
    ax2.set_ylabel("PDF", fontsize=8)
    ax2.tick_params(axis="both", labelsize=8)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=160)
    plt.close(fig2)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_report(
    project_name,
    engineer_name,
    title,
    base_df,
    df_calc,
    tol_rss,
    tol_wc,
    cpk_stack_rss,
    yield_stack_rss,
    sigma_stack_val,
    df_sigma_val,
    rss_factor_data_val,
    ta_loop_image_bytes,
) -> BytesIO:
    """建立 PDF 報告（Landscape A4），回傳 BytesIO 物件"""
    buffer = BytesIO()
    page_size = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size

    # ========== Page 1: 封面 + TA Loop ==========
    y = height - 15 * mm

    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, y, "Tolerance Analysis Report")
    y -= 12 * mm

    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Project : {ascii_only(project_name)}")
    y -= 5 * mm
    c.drawString(20 * mm, y, f"Engineer: {engineer_name}")
    y -= 5 * mm
    c.drawString(20 * mm, y, f"Title   : {ascii_only(title)}")
    y -= 10 * mm

    # TA Loop 圖（靠左）
    if ta_loop_image_bytes is not None:
        try:
            img = ImageReader(BytesIO(ta_loop_image_bytes))
            img_w, img_h = img.getSize()
            max_w = width - 40 * mm
            max_h = 80 * mm
            scale = min(max_w / img_w, max_h / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale

            c.setFont("Helvetica-Bold", 12)
            c.drawString(20 * mm, y, "TA Loop")
            y -= 6 * mm

            img_x = 20 * mm   # ⭐ 靠左排版
            c.drawImage(
                img,
                img_x,
                y - draw_h,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            y -= draw_h + 8 * mm
        except Exception:
            c.setFont("Helvetica", 9)
            c.drawString(20 * mm, y, "[TA Loop image error]")
            y -= 8 * mm

    c.showPage()

    # ========== Page 2: 1. Tolerance Setting + 2. Tolerance stack up (Worst case & RSS) ==========

    # ---- 1. Tolerance Setting ----
    y = height - 15 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "1. Tolerance Setting")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "Label")
    c.drawString(35 * mm, y, "Tolerance Loop")
    c.drawString(110 * mm, y, "Dimension")
    c.drawString(135 * mm, y, "Tol (±T)")
    c.drawString(155 * mm, y, "Cpk")
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    def new_page_tolerance(cont_title: str):
        nonlocal y
        c.showPage()
        y = height - 15 * mm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, y, cont_title)
        y -= 10 * mm
        c.setFont("Helvetica-Bold", 9)
        c.drawString(20 * mm, y, "Label")
        c.drawString(35 * mm, y, "Tolerance Loop")
        c.drawString(110 * mm, y, "Dimension")
        c.drawString(135 * mm, y, "Tol (±T)")
        c.drawString(155 * mm, y, "Cpk")
        y -= 6 * mm
        c.setFont("Helvetica", 9)

    for _, row in df_calc.iterrows():
        if y < 25 * mm:
            new_page_tolerance("1. Tolerance Setting (cont.)")
        label = ascii_only(row.get(COL_LABEL, ""))
        path = ascii_only(row.get(COL_PATH, ""))
        dim = "" if pd.isna(row.get(COL_DIM)) else f"{row[COL_DIM]:.3f}"
        tol = "" if pd.isna(row.get(COL_TOL)) else f"{row[COL_TOL]:.3f}"
        cpk_v = "" if pd.isna(row.get(COL_CPK)) else f"{row[COL_CPK]:.2f}"

        c.drawString(20 * mm, y, label)
        c.drawString(35 * mm, y, path[:60])
        c.drawString(110 * mm, y, dim)
        c.drawString(135 * mm, y, tol)
        c.drawString(155 * mm, y, cpk_v)
        y -= 5 * mm

    y -= 5 * mm

    # ---- 2. Tolerance stack up (Worst case & RSS) ----
    if y < 50 * mm:
        c.showPage()
        y = height - 15 * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "2. Tolerance stack up (Worst case & RSS)")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "Item")
    c.drawString(60 * mm, y, "Value")
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    def summary_row(label, value):
        nonlocal y
        c.drawString(20 * mm, y, label)
        c.drawString(60 * mm, y, value)
        y -= 5 * mm

    # ⭐ 順序改為：WC → RSS Tol → RSS Cpk → RSS Yield
    summary_row("WC  Tol (mm)", f"{tol_wc:.5f}")
    summary_row("RSS Tol (mm)", f"{tol_rss:.5f}")
    if not math.isnan(cpk_stack_rss):
        summary_row("RSS Cpk", f"{cpk_stack_rss:.3f}")
    if not math.isnan(yield_stack_rss):
        summary_row("RSS Yield (%)", f"{yield_stack_rss*100:.5f}%")

    y -= 6 * mm

    # xRSS sweep 1.0 ~ 1.5
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "xRSS")
    c.drawString(40 * mm, y, "Tol (mm)")
    c.drawString(80 * mm, y, "Cpk")
    c.drawString(110 * mm, y, "Yield%")
    c.drawString(145 * mm, y, "Defect Rate (PPM)")
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    factor = 1.0
    while factor <= 1.5001 + 1e-9:
        if y < 25 * mm:
            c.showPage()
            y = height - 15 * mm
            c.setFont("Helvetica-Bold", 14)
            c.drawString(20 * mm, y, "2. Tolerance stack up (Worst case & RSS) (cont.)")
            y -= 10 * mm
            c.setFont("Helvetica-Bold", 9)
            c.drawString(20 * mm, y, "xRSS")
            c.drawString(40 * mm, y, "Tol (mm)")
            c.drawString(80 * mm, y, "Cpk")
            c.drawString(110 * mm, y, "Yield%")
            c.drawString(145 * mm, y, "Defect Rate (PPM)")
            y -= 6 * mm
            c.setFont("Helvetica", 9)

        f_rounded = round(factor * 10) / 10.0
        tol_scaled = tol_rss * f_rounded
        cpk_scaled = cpk_stack_rss * f_rounded if not math.isnan(cpk_stack_rss) else float("nan")
        yld_scaled = central_yield_by_cpk(cpk_scaled) if not math.isnan(cpk_scaled) else float("nan")
        ppm_scaled = (1 - yld_scaled) * 1_000_000.0 if not math.isnan(yld_scaled) else float("nan")

        c.drawString(20 * mm, y, f"{f_rounded:.1f}")
        c.drawString(40 * mm, y, f"{tol_scaled:.5f}")
        if not math.isnan(cpk_scaled):
            c.drawString(80 * mm, y, f"{cpk_scaled:.3f}")
        if not math.isnan(yld_scaled):
            c.drawString(110 * mm, y, f"{yld_scaled*100:.5f}%")
        if not math.isnan(ppm_scaled):
            c.drawString(145 * mm, y, f"{ppm_scaled:.1f}")
        y -= 5 * mm

        factor += 0.1

    c.showPage()

    # ========== Page 3: 3. Tolerance stack up (Six sigma) + Normal Plot ==========

    y = height - 15 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "3. Tolerance stack up (Six sigma)")
    y -= 10 * mm

    if df_sigma_val is not None:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(20 * mm, y, "Sigma")
        c.drawString(40 * mm, y, "Tol Stack")
        c.drawString(80 * mm, y, "Yield%")
        c.drawString(120 * mm, y, "Defect Rate (PPM)")
        c.drawString(165 * mm, y, "Level / Remark")
        y -= 6 * mm
        c.setFont("Helvetica", 9)

        for _, row in df_sigma_val.iterrows():
            if y < 55 * mm:
                c.showPage()
                y = height - 15 * mm
                c.setFont("Helvetica-Bold", 14)
                c.drawString(20 * mm, y, "3. Tolerance stack up (Six sigma) (cont.)")
                y -= 10 * mm
                c.setFont("Helvetica-Bold", 9)
                c.drawString(20 * mm, y, "Sigma")
                c.drawString(40 * mm, y, "Tol Stack")
                c.drawString(80 * mm, y, "Yield%")
                c.drawString(120 * mm, y, "Defect Rate (PPM)")
                c.drawString(165 * mm, y, "Level / Remark")
                y -= 6 * mm
                c.setFont("Helvetica", 9)

            sigma_label = ascii_only(row["Sigma Level"])
            tol_stack_v = row["Tol Stack"]
            yld_str = ascii_only(row["理論良率 (Yield %)"])
            ppm_v = row["不良率 (PPM)"]

            level_raw = str(row["良率級別"] or "")
            level_eng = ascii_only(level_raw.split("\n")[-1])
            rm = ascii_only(row["備註"] or "")

            combined_lr = level_eng
            if rm:
                combined_lr += f" / {rm}"

            c.drawString(20 * mm, y, sigma_label)
            c.drawString(40 * mm, y, f"{tol_stack_v:.5f}")
            c.drawString(80 * mm, y, yld_str)
            c.drawString(120 * mm, y, f"{ppm_v:.1f}")
            c.drawString(165 * mm, y, combined_lr)
            y -= 5 * mm

    # Normal Plot（簡潔版）
    if y < 70 * mm:
        c.showPage()
        y = height - 15 * mm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, y, "3. Tolerance stack up (Six sigma) (Normal Plot)")
        y -= 10 * mm
    else:
        y -= 8 * mm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20 * mm, y, "Normal Plot")
        y -= 8 * mm

    normal_plot_bytes = generate_clean_normal_plot(sigma_stack_val, tol_wc)
    if normal_plot_bytes is not None:
        img_np = ImageReader(BytesIO(normal_plot_bytes))
        img_w, img_h = img_np.getSize()
        max_w = width - 40 * mm
        max_h = height - 40 * mm
        scale = min(max_w / img_w, max_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        img_x = (width - draw_w) / 2.0
        img_y = y - draw_h

        c.drawImage(
            img_np,
            img_x,
            img_y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


ta_loop_bytes = ta_loop_image.getvalue() if ta_loop_image is not None else None

if not df_calc.empty:
    pdf_buffer = build_pdf_report(
        project_name=project_name,
        engineer_name=engineer_name,
        title=title,
        base_df=base_df,
        df_calc=df_calc,
        tol_rss=tol_rss,
        tol_wc=tol_wc,
        cpk_stack_rss=cpk_stack_rss,
        yield_stack_rss=yield_stack_rss,
        sigma_stack_val=sigma_stack,
        df_sigma_val=df_sigma,
        rss_factor_data_val=rss_factor_data,
        ta_loop_image_bytes=ta_loop_bytes,
    )

    if title and title.strip():
        safe_title = title.strip().replace("/", "_").replace("\\", "_")
        file_name = f"{safe_title}.pdf"
    else:
        file_name = "TA_Report.pdf"

    st.download_button(
        label="📄 下載完整 PDF 報告",
        data=pdf_buffer,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    """
使用方式：
• 在上方輸入 Project / Engineer / Title。
• 在表格輸入各零件的設計尺寸、公差與 Cpk，左側代號會自動產生。
• 若需刪除列，勾選「刪除」再按 🗑。
• 可用 RSS 倍率試算不同良率。
• 所有計算會即時更新，並可匯出 PDF 報告。
"""
)
