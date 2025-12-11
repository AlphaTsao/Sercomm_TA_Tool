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
    # ★ 新增：計算 Defect Rate (PPM)
    defect_ppm_scaled = (1.0 - yield_scaled) * 1_000_000.0

    df_rss = pd.DataFrame(
        {
            "x RSS": [f"{rss_factor:.1f}"],
            "Tol (mm)": [f"{rss_tol_scaled:.5f}"],
            "Yield Rate (%)": [f"{yield_scaled * 100.0:.5f}%"],
            "Defect Rate (PPM)": [f"{defect_ppm_scaled:.1f}"],
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

import base64
from io import BytesIO

st.subheader("6️⃣ 匯出 PDF 報告")
st.caption("將目前頁面上的 TA 資訊完整匯出為 PDF 報告（橫向），並自動加上 Sercomm 浮水印。")

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 避免 PNG 有一點壞掉就整個報錯

# Sercomm logo 的 Base64 字串（去掉 data:image/png;base64, 前綴）
SERCOMM_LOGO_BASE64 = """
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
""".strip()


def ascii_only(s: str) -> str:
    """只保留 ASCII 字元（避免中文在某些 PDF viewer 變黑框）"""
    return "".join(ch for ch in str(s) if ord(ch) < 128)


# ---------- Normal Plot（PDF 專用，與畫面版 X 軸同步） ----------
def generate_clean_normal_plot(sigma_stack_val, tol_wc_val):
    """產生 PDF 報告用 Normal Plot，X 軸刻度與畫面版一致，背景透明"""
    if sigma_stack_val <= 0 or tol_wc_val == 0:
        return None

    # 與畫面版相同邏輯：由 WC 公差與 sigma 算出可視範圍
    k_max_local = min(6.0, abs(tol_wc_val) / sigma_stack_val)
    if k_max_local <= 0:
        return None

    x_min_local = -k_max_local * sigma_stack_val
    x_max_local = k_max_local * sigma_stack_val

    # 畫面版 X 軸刻度：固定 9 個 ticks
    num_ticks = 9
    x_ticks = np.linspace(x_min_local, x_max_local, num_ticks)

    # 曲線取樣點與 PDF
    x_values = np.linspace(x_min_local, x_max_local, 420)
    y_values = [normal_pdf(x, mu=0.0, sigma=sigma_stack_val) for x in x_values]
    y_max = max(y_values) if y_values else 0.0

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.0))

    ax2.plot(
        x_values,
        y_values,
        linewidth=0.8,
        color="#1a76d2",
        zorder=3,
    )

    # X 軸刻度與畫面完全一致（位置、數值、小數 3 位）
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{x:.3f}" for x in x_ticks], fontsize=8)

    ax2.set_xlabel("Tolerance (mm)", fontsize=8)
    ax2.set_ylabel("PDF", fontsize=8)
    ax2.tick_params(axis="both", which="major", labelsize=8)

    if y_max > 0:
        ax2.set_ylim(bottom=0, top=y_max * 1.09)
    else:
        ax2.set_ylim(bottom=0)

    # 簡潔風格：無上、右框
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 背景透明，讓浮水印從後面透出
    fig2.patch.set_alpha(0.0)
    ax2.set_facecolor("none")

    plt.tight_layout(pad=0.25)

    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=160, transparent=True)
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
    sercomm_logo_bytes,
) -> BytesIO:
    """建立 PDF 報告（Landscape A4），回傳 BytesIO 物件"""

    buffer = BytesIO()
    page_size = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size

    # ---------- 準備 Sercomm 浮水印圖 ----------
    logo_reader = None
    if sercomm_logo_bytes:
        try:
            img_logo = Image.open(BytesIO(sercomm_logo_bytes)).convert("RGBA")
            img_logo.load()

            # 只保留藍色字（用 HSV 過濾），底線變透明
            img_rgb = img_logo.convert("RGB")
            img_hsv = img_rgb.convert("HSV")
            h, s, v = img_hsv.split()
            h_arr = np.array(h, dtype=np.uint8)
            s_arr = np.array(s, dtype=np.uint8)
            v_arr = np.array(v, dtype=np.uint8)

            mask = (
                (s_arr > 60)
                & (v_arr > 50)
                & (h_arr >= 120)
                & (h_arr <= 210)
            )
            alpha_arr = np.where(mask, 255, 0).astype("uint8")
            alpha = Image.fromarray(alpha_arr, mode="L")

            # 整體變淡（≈ 35% 原色，看起來像半透明）
            white = Image.new("RGB", img_rgb.size, (255, 255, 255))
            light_rgb = Image.blend(white, img_rgb, 0.35)
            img_final = Image.merge("RGBA", (*light_rgb.split(), alpha))

            buf_logo = BytesIO()
            img_final.save(buf_logo, format="PNG")
            buf_logo.seek(0)
            logo_reader = ImageReader(buf_logo)
        except Exception:
            logo_reader = None

    def draw_watermark():
        """在當前頁面畫 Sercomm 浮水印（置中、45° 旋轉、放在最底層）"""
        if not logo_reader:
            return
        c.saveState()
        img_w, img_h = logo_reader.getSize()

        target_w = width * 0.55
        scale = target_w / float(img_w)
        draw_w = img_w * scale
        draw_h = img_h * scale

        c.translate(width / 2.0, height / 2.0)
        c.rotate(45)
        c.drawImage(
            logo_reader,
            -draw_w / 2.0,
            -draw_h / 2.0,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.restoreState()

    # =================== Page 1: 封面 + TA Loop ===================
    draw_watermark()

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

    # TA Loop 圖（靠左顯示）
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

            img_x = 20 * mm
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

    # =================== Page 2: Tolerance Setting ===================
    draw_watermark()

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
        draw_watermark()
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
        label = ascii_only(row.get("公差路徑代號", ""))
        path = ascii_only(row.get("公差路徑 (Tolerance Loop)", ""))
        dim = "" if pd.isna(row.get("Dimension (mm)")) else f"{row['Dimension (mm)']:.3f}"
        tol_val = "" if pd.isna(row.get("公差(±T)")) else f"{row['公差(±T)']:.3f}"
        cpk_v = "" if pd.isna(row.get("CPK")) else f"{row['CPK']:.2f}"

        c.drawString(20 * mm, y, label)
        c.drawString(35 * mm, y, path[:60])
        c.drawString(110 * mm, y, dim)
        c.drawString(135 * mm, y, tol_val)
        c.drawString(155 * mm, y, cpk_v)
        y -= 5 * mm

    y -= 5 * mm

    # =================== Page 2 (續): Stack up summary + xRSS 表 ===================
    if y < 50 * mm:
        c.showPage()
        draw_watermark()
        y = height - 15 * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "2. Tolerance stack up (Worst case & RSS)")
    y -= 10 * mm

    # Summary block
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "Item")
    c.drawString(60 * mm, y, "Value")
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    def summary_row(lbl, val):
        nonlocal y
        c.drawString(20 * mm, y, lbl)
        c.drawString(60 * mm, y, val)
        y -= 5 * mm

    summary_row("WC  Tol (mm)", f"{tol_wc:.5f}")
    summary_row("RSS Tol (mm)", f"{tol_rss:.5f}")
    if not math.isnan(cpk_stack_rss):
        summary_row("RSS Cpk", f"{cpk_stack_rss:.3f}")
    if not math.isnan(yield_stack_rss):
        summary_row("RSS Yield (%)", f"{yield_stack_rss*100:.5f}%")

    y -= 6 * mm

    # xRSS sweep table（含 Defect Rate PPM）
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
            draw_watermark()
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

    # =================== Page 3: Six sigma 表 + Normal Plot ===================
    draw_watermark()

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
                draw_watermark()
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

            s_label = ascii_only(row["Sigma Level"])
            tol_v = row["Tol Stack"]
            yld_str = ascii_only(row["理論良率 (Yield %)"])
            ppm_v = row["不良率 (PPM)"]

            level_raw = str(row["良率級別"] or "")
            level_eng = ascii_only(level_raw.split("\n")[-1])
            rm = ascii_only(row["備註"] or "")
            combined_lr = level_eng if not rm else f"{level_eng} / {rm}"

            c.drawString(20 * mm, y, s_label)
            c.drawString(40 * mm, y, f"{tol_v:.5f}")
            c.drawString(80 * mm, y, yld_str)
            c.drawString(120 * mm, y, f"{ppm_v:.1f}")
            c.drawString(165 * mm, y, combined_lr)
            y -= 5 * mm

    # ---------- Normal Plot（簡潔版，與畫面版 X 軸一致，盡量放大） ----------
    if y < 70 * mm:
        c.showPage()
        draw_watermark()
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

        max_w = width - 40 * mm   # 左右預留 20mm
        max_h = height - 40 * mm  # 上下總共預留約 40mm
        base_scale = min(max_w / img_w, max_h / img_h)
        scale = base_scale * 0.98  # 稍微縮 2%，避免貼邊或跳頁

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

    # 最後一頁直接收尾（不再多 showPage）
    c.save()
    buffer.seek(0)
    return buffer


# ---------- 觸發產生並下載 PDF 報告 ----------
ta_loop_bytes = ta_loop_image.getvalue() if ta_loop_image is not None else None

# 讀取 Sercomm logo 圖檔，若沒有檔案就用 Base64
try:
    with open("sercomm_logo.png", "rb") as f:
        sercomm_logo_bytes = f.read()
except Exception:
    try:
        sercomm_logo_bytes = base64.b64decode(SERCOMM_LOGO_BASE64)
    except Exception:
        sercomm_logo_bytes = None

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
        sercomm_logo_bytes=sercomm_logo_bytes,
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
