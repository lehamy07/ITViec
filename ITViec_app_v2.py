"""ITViec Company Recommender & Insights - bản v2 (cải tiến UX/UI).

Logic mô hình giữ nguyên bản gốc ITViec_app.py. Những chỗ khác bản gốc:
  - Giao diện: theme qua .streamlit/config.toml, CSS gọn, thẻ công ty, chip, thanh điểm.
  - Điều hướng: nút nhảy giữa các trang (từ thẻ công ty sang đánh giá / công ty tương tự).
  - SỬA: tên cột sai (Company_Type, Overtime_Policy, working_here) -> tên cột thật.
"""
import ast
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import prepare_data, predict_with_class0_threshold

st.set_page_config(
    page_title="ITViec Company Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Màu sắc nằm ở .streamlit/config.toml; CSS chỉ chỉnh những gì theme không làm được.
st.markdown(
    """
<style>
.block-container {max-width: 1200px; padding-top: 2.5rem;}
.hero {background: linear-gradient(135deg,#1E3A8A 0%,#2563EB 100%); color:#fff;
       border-radius:16px; padding:36px 40px; margin-bottom:24px;}
.hero h1 {color:#fff !important; margin:0 0 8px 0; font-size:2.1rem;}
.hero p {color:#DBEAFE; margin:0; font-size:1.05rem;}
.chip {display:inline-block; background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE;
       border-radius:999px; padding:2px 12px; margin:0 6px 6px 0; font-size:0.85rem;}
.chip.warn {background:#FFF7ED; color:#C2410C; border-color:#FED7AA;}
.verdict {border-radius:14px; padding:20px 24px; font-size:1.25rem; font-weight:700;}
.verdict.ok {background:#DCFCE7; color:#166534; border:1px solid #86EFAC;}
.verdict.no {background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5;}
.verdict small {display:block; font-weight:400; font-size:0.95rem; margin-top:4px;}
.step {background:#F1F5F9; border-radius:12px; padding:16px 18px; height:100%;}
.step b {color:#2563EB;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 1. TẢI MÔ HÌNH VÀ DỮ LIỆU
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ITviec_models1"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_models():
    rec = joblib.load(MODEL_DIR / "company_recommender_bundle.joblib")
    clf = joblib.load(MODEL_DIR / "best_logistic_model.pkl")
    return rec, clf


@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_excel(name):
    return pd.read_excel(DATA_DIR / name)


try:
    rec_model, clf_bundle = load_models()
    overview_reviews = load_excel("Overview_Reviews.xlsx")
except (FileNotFoundError, ValueError, TypeError, KeyError, OSError) as error:
    st.error(f"Không thể tải mô hình hoặc dữ liệu: {error}")
    st.stop()

vectorizer = rec_model["vectorizer"]
tfidf_matrix = rec_model["tfidf_matrix"]
similarity_matrix = rec_model["similarity_matrix"]
companies_df = rec_model["companies_df"].reset_index(drop=True)
clf_model = clf_bundle["pipeline"]
clf_threshold = clf_bundle["best_threshold_class0"]

if (
    len(companies_df) != tfidf_matrix.shape[0]
    or similarity_matrix.shape != (len(companies_df),) * 2
):
    st.error("Recommendation Model không hợp lệ.")
    st.stop()

COMPANY_NAMES = sorted(companies_df["Company Name"].dropna().unique().tolist())

# =========================================================
# 2. HÀM HỖ TRỢ
# =========================================================


def safe_text(value, default="Chưa cập nhật"):
    if value is None or pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


def truncate_text(value, max_words):
    words = safe_text(value, "Chưa có thông tin.").split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def chips(values, warn=False):
    """Hiển thị danh sách nhãn nhỏ (bỏ giá trị trống)."""
    cls = "chip warn" if warn else "chip"
    items = [v for v in values if safe_text(v, "") not in ("", "Unknown")]
    if not items:
        return ""
    return "".join(f'<span class="{cls}">{html.escape(str(v))}</span>' for v in items)


def cities(location):
    """Cột Location lưu dạng chuỗi dict {'Ha Noi': [...]} -> 'Ha Noi, Ho Chi Minh'."""
    try:
        return list(ast.literal_eval(str(location)).keys())
    except (ValueError, SyntaxError, AttributeError):
        return []


def get_company(name):
    result = companies_df[companies_df["Company Name"] == name]
    return None if result.empty else result.iloc[0]


def itviec_url(company):
    url = safe_text(company.get("Href"), "")
    return url if url.startswith("http") else None


def go(page, **state):
    """Callback nút bấm: đổi trang và gán sẵn giá trị cho widget ở trang đích."""
    st.session_state["page"] = page
    st.session_state.update(state)


def top_k(scores, top_n, exclude=None):
    scores = np.asarray(scores, dtype=float).copy()
    if exclude is not None:
        scores[exclude] = -1.0
    idx = np.argsort(scores)[::-1][:top_n]
    result = companies_df.iloc[idx].copy()
    result["similarity"] = scores[idx] * 100
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def recommend_by_content(text, top_n):
    sims = cosine_similarity(vectorizer.transform([text]), tfidf_matrix).flatten()
    return top_k(sims, top_n)


def recommend_similar(name, top_n):
    matches = np.flatnonzero(companies_df["Company Name"].to_numpy() == name)
    if len(matches) == 0:
        return pd.DataFrame()
    return top_k(similarity_matrix[matches[0]], top_n, exclude=matches[0])


# Tên trang (dùng làm key điều hướng)
P_HOME, P_REC, P_REV, P_ABOUT = (
    "🏠 Trang chủ",
    "🔎 Tìm công ty",
    "⭐ Đánh giá công ty",
    "ℹ️ Giới thiệu",
)
PAGES = [P_HOME, P_REC, P_REV, P_ABOUT]
BY_NEED, BY_NAME = "Theo nhu cầu", "Theo tên công ty"

# =========================================================
# 3. THÀNH PHẦN GIAO DIỆN DÙNG CHUNG
# =========================================================


def company_header(company):
    """Tên công ty + nhãn thông tin + liên kết sang ITViec."""
    st.markdown(f"### {safe_text(company.get('Company Name'))}")
    st.markdown(
        chips(
            [
                company.get("Company industry"),
                company.get("Company Type"),
                company.get("Company size"),
                company.get("Country"),
                *cities(company.get("Location")),
            ]
        ),
        unsafe_allow_html=True,
    )


def company_card(row, prefix):
    """Thẻ kết quả đề xuất: điểm, nhãn, mô tả ngắn và các nút hành động."""
    name = safe_text(row.get("Company Name"))
    sim = float(row.get("similarity", 0))
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"### {int(row['rank'])}. {name}")
        with right:
            st.metric("Độ tương đồng", f"{sim:.0f}/100")
        st.progress(float(np.clip(sim / 100, 0, 1)))
        st.markdown(
            chips(
                [
                    row.get("Company industry"),
                    row.get("Company Type"),
                    row.get("Company size"),
                    row.get("Country"),
                    *cities(row.get("Location")),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.write(truncate_text(row.get("Company overview"), 45))
        skills = safe_text(row.get("Our key skills"), "")
        if skills:
            st.caption(f"**Kỹ năng:** {truncate_text(skills, 25)}")

        c1, c2, c3 = st.columns(3)
        c1.button(
            "⭐ Xem đánh giá",
            key=f"{prefix}_rev_{row['id']}",
            width="stretch",
            on_click=go,
            args=(P_REV,),
            kwargs={"review_company": name},
        )
        c2.button(
            "🧭 Công ty tương tự",
            key=f"{prefix}_sim_{row['id']}",
            width="stretch",
            on_click=go,
            args=(P_REC,),
            kwargs={"rec_method": BY_NAME, "rec_company": name},
        )
        url = itviec_url(row)
        if url:
            c3.link_button("🔗 Trang ITViec", url, width="stretch")


def render_results(result, prefix):
    for _, row in result.iterrows():
        company_card(row, prefix)


# =========================================================
# 4. SIDEBAR
# =========================================================
st.session_state.setdefault("page", P_HOME)

with st.sidebar:
    st.title("💼 IT Company")
    st.caption("Recommender & Insights")
    st.radio("Điều hướng", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption(f"Dữ liệu: **{len(companies_df)}** công ty IT")
    with st.expander("Thông tin nhóm"):
        st.markdown(
            "**Bùi Thị Thư**  \nbuithithu.ntt@gmail.com\n\n"
            "**Lê Thị Hà My**  \nlehamy.yds@gmail.com"
        )

page = st.session_state["page"]

# =========================================================
# 5. TRANG CHỦ
# =========================================================
if page == P_HOME:
    st.markdown(
        """
<div class="hero">
  <h1>Tìm công ty IT phù hợp với bạn</h1>
  <p>Gợi ý công ty theo kỹ năng và nhu cầu, xem điểm đánh giá từ nhân viên
  và thử dự đoán một review là Recommend hay Not Recommend.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Công ty trong hệ thống", f"{len(companies_df):,}")
    m2.metric("Công ty có điểm đánh giá", f"{overview_reviews['id'].nunique():,}")
    m3.metric("Quốc gia", f"{companies_df['Country'].nunique():,}")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1.container(border=True):
        st.subheader("🔎 Tìm công ty")
        st.write(
            "Nhập kỹ năng, vị trí hoặc lĩnh vực bạn quan tâm, hoặc chọn một công ty "
            "bạn thích để tìm những công ty tương tự."
        )
        st.caption("TF-IDF · Cosine Similarity")
        st.button(
            "Bắt đầu tìm",
            type="primary",
            width="stretch",
            key="home_rec",
            on_click=go,
            args=(P_REC,),
        )
    with c2.container(border=True):
        st.subheader("⭐ Đánh giá công ty")
        st.write(
            "Xem thông tin công ty, điểm đánh giá từ nhân viên và thử dự đoán "
            "một review có xu hướng Recommend hay Not Recommend."
        )
        st.caption("Logistic Regression · Employee Insights")
        st.button(
            "Xem đánh giá",
            type="primary",
            width="stretch",
            key="home_rev",
            on_click=go,
            args=(P_REV,),
        )

    st.write("")
    st.subheader("Cách dùng nhanh")
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="step"><b>1. Tìm</b><br>Gõ kỹ năng (Python, SQL, Fintech...) '
        "hoặc chọn một công ty đã biết.</div>",
        unsafe_allow_html=True,
    )
    s2.markdown(
        '<div class="step"><b>2. So sánh</b><br>Xem độ tương đồng, lĩnh vực, '
        "quy mô và kỹ năng của từng công ty.</div>",
        unsafe_allow_html=True,
    )
    s3.markdown(
        '<div class="step"><b>3. Đánh giá</b><br>Mở điểm đánh giá của nhân viên '
        "hoặc thử viết một review để mô hình dự đoán.</div>",
        unsafe_allow_html=True,
    )

# =========================================================
# 6. TÌM CÔNG TY
# =========================================================
elif page == P_REC:
    st.title("🔎 Tìm công ty")
    st.caption("Gợi ý công ty dựa trên nội dung mô tả (Content-Based).")

    st.session_state.setdefault("rec_method", BY_NEED)
    method = st.segmented_control(
        "Phương thức",
        [BY_NEED, BY_NAME],
        key="rec_method",
        label_visibility="collapsed",
    ) or BY_NEED

    top_n = st.slider("Số công ty hiển thị", 1, 10, 5, key="rec_top_n")

    if method == BY_NEED:
        st.caption("Gợi ý nhanh (từ khóa tiếng Anh cho kết quả tốt hơn):")
        examples = ["Python", "Data Analyst", "Java Backend", "Fintech", "E-commerce", "AI Machine Learning"]
        for col, ex in zip(st.columns(len(examples)), examples):
            col.button(
                ex,
                key=f"ex_{ex}",
                width="stretch",
                on_click=lambda t=ex: st.session_state.update(rec_query=t, rec_go=True),
            )

        query = st.text_area(
            "Bạn đang tìm kiếm điều gì?",
            placeholder="Ví dụ: Data Analyst, Python, Power BI, good working environment...",
            height=100,
            key="rec_query",
        )
        clicked = st.button("Tìm công ty phù hợp", type="primary", width="stretch", key="rec_search")

        if clicked or st.session_state.pop("rec_go", False):
            if not query.strip():
                st.warning("Vui lòng nhập nhu cầu tìm kiếm.")
            else:
                st.session_state["rec_result"] = (query, recommend_by_content(query, top_n))

        saved = st.session_state.get("rec_result")
        if saved:
            st.divider()
            st.success(f"Top {len(saved[1])} công ty phù hợp với: **{saved[0]}**")
            render_results(saved[1], "need")

    else:  # BY_NAME: kết quả cập nhật ngay khi đổi công ty, không cần bấm nút
        name = st.selectbox(
            "Chọn công ty bạn quan tâm (gõ để tìm)",
            COMPANY_NAMES,
            key="rec_company",
        )
        company = get_company(name)
        if company is not None:
            with st.container(border=True):
                company_header(company)
                st.write(truncate_text(company.get("Company overview"), 60))

        st.divider()
        result = recommend_similar(name, top_n)
        if result.empty:
            st.info("Không tìm thấy công ty tương tự.")
        else:
            st.success(f"{len(result)} công ty tương tự với **{name}**")
            render_results(result, "name")

# =========================================================
# 7. ĐÁNH GIÁ CÔNG TY
# =========================================================
elif page == P_REV:
    st.title("⭐ Đánh giá công ty")
    st.caption("Xem thông tin, điểm đánh giá của nhân viên và thử phân loại một review.")

    name = st.selectbox(
        "Chọn công ty (gõ để tìm)", COMPANY_NAMES, key="review_company"
    )
    company = get_company(name)
    if company is None:
        st.warning("Không tìm thấy thông tin công ty.")
        st.stop()

    with st.container(border=True):
        company_header(company)
        a, b, c = st.columns([1, 1, 1])
        a.caption("Ngày làm việc")
        a.write(safe_text(company.get("Working days")))
        b.caption("Chính sách làm thêm")
        b.write(safe_text(company.get("Overtime Policy")))
        with c:
            url = itviec_url(company)
            if url:
                st.link_button("🔗 Xem trên ITViec", url, width="stretch")
            st.button(
                "🧭 Công ty tương tự",
                key="rev_similar",
                width="stretch",
                on_click=go,
                args=(P_REC,),
                kwargs={"rec_method": BY_NAME, "rec_company": name},
            )

    tab_info, tab_score, tab_try = st.tabs(
        ["📄 Giới thiệu", "📊 Điểm đánh giá", "🤖 Thử dự đoán review"]
    )

    # ---- Tab 1: giới thiệu ----
    with tab_info:
        # SỬA: tên cột đúng là "Why you'll love working here" (bản gốc ghi working_here nên không bao giờ hiện)
        sections = [
            ("Tổng quan công ty", "Company overview"),
            ("Kỹ năng chính", "Our key skills"),
            ("Vì sao nên làm việc tại đây", "Why you'll love working here"),
        ]
        shown = False
        for title, key in sections:
            text = safe_text(company.get(key), "")
            if text:
                shown = True
                st.markdown(f"**{title}**")
                st.write(text)
        if not shown:
            st.info("Chưa có mô tả cho công ty này.")

    # ---- Tab 2: điểm đánh giá ----
    with tab_score:
        rows = overview_reviews[overview_reviews["id"] == company["id"]]
        stats = rows.iloc[0] if not rows.empty else None
        overall = pd.to_numeric(stats.get("Overall rating"), errors="coerce") if stats is not None else np.nan

        if stats is None or pd.isna(overall) or overall <= 0:
            st.info("Chưa có điểm đánh giá cho công ty này.")
        else:
            n_reviews = pd.to_numeric(stats.get("Number of reviews"), errors="coerce")
            top, mid = st.columns(2)
            top.metric("Điểm tổng thể", f"{overall:.1f}/5")
            mid.metric("Số lượt đánh giá", f"{int(n_reviews):,}" if pd.notna(n_reviews) and n_reviews > 0 else "N/A")

            criteria = [
                ("Lương & phúc lợi", "Salary & benefits"),
                ("Đào tạo & học tập", "Training & learning"),
                ("Quan tâm từ quản lý", "Management cares about me"),
                ("Văn hóa & giải trí", "Culture & fun"),
                ("Môi trường làm việc", "Office & workspace"),
            ]
            st.write("")
            for col, (label, key) in zip(st.columns(5), criteria):
                value = pd.to_numeric(stats.get(key), errors="coerce")
                col.metric(label, f"{value:.1f}/5" if pd.notna(value) and value > 0 else "N/A")

            rec = stats.get("Recommend working here to a friend")
            if isinstance(rec, str):
                rec = rec.strip().replace("%", "")
            rec = pd.to_numeric(rec, errors="coerce")
            if pd.notna(rec):
                rec = float(np.clip(rec * 100 if 0 < rec <= 1 else rec, 0, 100))
                st.write("")
                st.markdown(f"**{rec:.0f}%** nhân viên sẵn sàng giới thiệu công ty cho bạn bè")
                st.progress(rec / 100)

    # ---- Tab 3: dự đoán review (form: chỉ chạy khi bấm nút) ----
    with tab_try:
        st.caption(
            "Chấm điểm và viết review, mô hình sẽ dự đoán review đó có xu hướng "
            f"giới thiệu **{name}** hay không."
        )
        rating_fields = [
            ("review_rating", "Mức độ hài lòng chung"),
            ("review_salary", "Lương & phúc lợi"),
            ("review_training", "Đào tạo & phát triển"),
            ("review_management", "Sự quan tâm từ quản lý"),
            ("review_culture", "Văn hóa & đồng nghiệp"),
            ("review_workspace", "Không gian làm việc"),
        ]
        with st.form("review_form"):
            ratings = {}
            for col, (key, label) in zip(st.columns(3) * 2, rating_fields):
                ratings[key] = col.slider(label, 1, 5, 4, key=f"rate_{key}")

            title = st.text_input("Tiêu đề", placeholder="Ví dụ: Môi trường làm việc tốt")
            l, r = st.columns(2)
            liked = l.text_area("Điều bạn thích", placeholder="Ví dụ: Đồng nghiệp thân thiện...", height=110)
            suggestions = r.text_area(
                "Điều bạn muốn cải thiện", placeholder="Ví dụ: Cải thiện chính sách nghỉ phép...", height=110
            )
            submitted = st.form_submit_button("Dự đoán review", type="primary", width="stretch")

        if submitted:
            if not any(t.strip() for t in (title, liked, suggestions)):
                st.warning("Vui lòng nhập ít nhất một nội dung đánh giá.")
            else:
                review_input = pd.DataFrame(
                    [
                        {
                            "Company Name": name,
                            # SỬA: bản gốc dùng "Company_Type"/"Overtime_Policy" (không tồn tại) nên luôn là NaN
                            "Company Type": company.get("Company Type"),
                            "Overtime Policy": company.get("Overtime Policy"),
                            "Rating": ratings["review_rating"],
                            "Salary & benefits": ratings["review_salary"],
                            "Training & learning": ratings["review_training"],
                            "Management cares about me": ratings["review_management"],
                            "Culture & fun": ratings["review_culture"],
                            "Office & workspace": ratings["review_workspace"],
                            "Title": title,
                            "What I liked": liked,
                            "Suggestions for improvement": suggestions,
                        }
                    ]
                )
                try:
                    detail, _ = predict_with_class0_threshold(
                        prepare_data(review_input), clf_model, clf_threshold
                    )
                    res = detail.iloc[0]
                    is_rec = int(res["is_recommend"]) == 1
                    p_rec = float(res["prob_class1_recommend"])
                except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                    st.error(f"Không thể phân loại review: {error}")
                else:
                    if is_rec:
                        st.markdown(
                            '<div class="verdict ok">✅ RECOMMEND'
                            f"<small>Review có xu hướng giới thiệu {html.escape(name)}.</small></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="verdict no">⚠️ NOT RECOMMEND'
                            f"<small>Review có xu hướng không giới thiệu {html.escape(name)}.</small></div>",
                            unsafe_allow_html=True,
                        )
                    st.write("")
                    p1, p0 = st.columns(2)
                    p1.metric("Xác suất Recommend", f"{p_rec * 100:.1f}%")
                    p1.progress(float(np.clip(p_rec, 0, 1)))
                    p0.metric("Xác suất Not Recommend", f"{(1 - p_rec) * 100:.1f}%")
                    p0.progress(float(np.clip(1 - p_rec, 0, 1)))
                    st.caption("Đây là dự đoán của mô hình, chỉ mang tính tham khảo.")

# =========================================================
# 8. GIỚI THIỆU
# =========================================================
else:
    st.title("ℹ️ Giới thiệu dự án")
    st.write(
        "Hệ thống hỗ trợ tìm kiếm, khám phá và đánh giá các công ty công nghệ dựa trên dữ liệu ITViec."
    )

    with st.container(border=True):
        st.subheader("1. Company Recommendation")
        st.write(
            "Content-Based Recommendation: biểu diễn mô tả công ty bằng TF-IDF rồi dùng "
            "Cosine Similarity để tìm các công ty có nội dung tương đồng."
        )

    with st.container(border=True):
        st.subheader("2. Review Classification")
        st.write(
            "Logistic Regression dự đoán xu hướng Recommend / Not Recommend từ điểm số và nội dung review. "
            "Đây là mô hình tốt nhất sau khi thử 4 mô hình (Logistic Regression, Linear SVM, KNN, Random Forest) "
            "trong Scikit-Learn."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", "0.7938")
        m2.metric("Recall", "0.8213")
        m3.metric("F1-Score", "0.7681")
        m4.metric("Ngưỡng Class 0", f"{clf_threshold:.2f}")

    st.info("Kết quả phân loại là dự đoán của mô hình, chỉ mang tính tham khảo.")

st.divider()
st.caption("IT Company Recommender System · Data-driven insights for IT career exploration")
