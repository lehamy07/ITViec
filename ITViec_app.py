from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_review_a, prepare_data, predict_with_class0_threshold

# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="ITViec Company Recommender & Insights",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1.1. CUSTOM CSS - GIAO DIỆN TỔNG THỂ
# ---------------------------------------------------------

st.markdown("""
<style>
   /* ===============================
       1. BẢNG MÀU CHỦ ĐẠO
    =============================== */
    :root {
        --primary-color: #2563EB;
        --text-color: #1E293B;
        --secondary-text: #475569;
        --border-color: #E2E8F0;
        --background-color: #FFFFFF;
        --hover-color: #EFF6FF;
    }

    /* ===============================
       2. NỀN ỨNG DỤNG
    =============================== */
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ===============================
       3. TIÊU ĐỀ
    =============================== */
    h1 {
        color: #172554 !important;
        font-weight: 750 !important;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #1E3A5F !important;
        font-weight: 700 !important;
    }

    h3, h4 {
        color: #334155 !important;
        font-weight: 650 !important;
    }

    /* ===============================
       4. NỘI DUNG VĂN BẢN
    =============================== */
    p, label,
    [data-testid="stCaption"],
    .stMarkdown {
        color: #334155;
    }

    [data-testid="stCaption"] {
        color: #475569 !important;
    }

    /* Chữ trong các thành phần nội dung */
    .stApp,
    .stApp div,
    .stApp span {
        --text-color: #1E293B;
    }

    /* ===============================
       5. SIDEBAR
    =============================== */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1E3A5F !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: #334155 !important;
    }

    /* Radio menu */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #334155 !important;
    }

    /* ===============================
       6. NÚT BẤM
    =============================== */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
        border: 1px solid #CBD5E1;
        color: #1E293B;
        background-color: #FFFFFF;
    }

    .stButton > button:hover {
        border-color: #2563EB !important;
        color: #1D4ED8 !important;
        background-color: #EFF6FF !important;
    }

    /* Nút chính */
    .stButton > button[kind="primary"] {
        background-color: #F8FAFC !important;
        border: 1px solid #2563EB !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* ===============================
       7. SLIDER - NÚT KÉO MÀU XANH
    =============================== */
    .stSlider [role="slider"] {
        background-color: #2563EB !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 0 2px #2563EB !important;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #2563EB !important;
    }

    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        background-color: #2563EB !important;
    }

    /* Giá trị và nhãn slider */
    .stSlider label,
    .stSlider [data-testid="stThumbValue"] {
        color: #1E293B !important;
    }

    /* ===============================
       8. Ô NHẬP LIỆU
    =============================== */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #1E293B !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }

    /* Placeholder */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #64748B !important;
        opacity: 1;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #CBD5E1;
        color: #1E293B;
    }

    /* ===============================
       9. THẺ NỘI DUNG
    =============================== */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E2E8F0;
        border-radius: 12px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border-color: #E2E8F0;
        border-radius: 8px;
    }

    /* ===============================
       10. METRIC VÀ KẾT QUẢ
    =============================== */
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
        font-weight: 700;
    }

    /* ===============================
       11. ĐƯỜNG VIỀN VÀ THÔNG BÁO
    =============================== */
    hr {
        border-color: #E2E8F0 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Chữ trong thông báo */
    [data-testid="stAlert"] p {
        color: #1E293B !important;
    }

    /* ===============================
       12. BẢNG DỮ LIỆU
    =============================== */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)  

# =========================================================
# 2. LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ITviec_models1"
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def load_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return joblib.load(path)

try:
    rec_model = load_file(MODEL_DIR / "company_recommender_bundle.joblib")
    clf_bundle = load_file(MODEL_DIR / "best_logistic_model.pkl")

except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
    st.error(f"Không thể tải mô hình: {error}")
    st.stop()
    
# ---------------------------------------------------------
# 2.1. Các thành phần của Recommendation Model
# ---------------------------------------------------------

vectorizer = rec_model["vectorizer"]
tfidf_matrix = rec_model["tfidf_matrix"]
similarity_matrix = rec_model["similarity_matrix"]
companies_df = rec_model["companies_df"]

# Kiểm tra Recommendation Model
if (
    len(companies_df) != tfidf_matrix.shape[0] 
    or similarity_matrix.shape != (len(companies_df), len(companies_df))
):
    st.error("Recommendation Model không hợp lệ.")
    st.stop()

# ---------------------------------------------------------
# 2.2. Các thành phần của Classification Model
# ---------------------------------------------------------   

clf_model = clf_bundle["pipeline"]
clf_threshold = clf_bundle["best_threshold_class0"]
clf_features = list(clf_bundle["feature_columns"])

# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_excel(path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    return pd.read_excel(path)

try:
    overview_reviews = load_excel(DATA_DIR / "Overview_Reviews.xlsx")
    reviews = load_excel(DATA_DIR / "Reviews.xlsx")
    overview_companies = load_excel(DATA_DIR / "Overview_Companies.xlsx")

except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Không thể tải dữ liệu: {error}")
    st.stop()
    
# =========================================================
# 4. THÔNG TIN CƠ BẢN VỀ DATASET
# =========================================================

# Recommendation model hiện tại được xây dựng trên 478 công ty.
# Chuẩn bị danh sách công ty dùng chung

company_options = (
    companies_df[["id", "Company Name"]]
    .sort_values("Company Name", na_position="last")
    .reset_index(drop=True)
)

# =========================================================
# 5. CÁC HÀM HỖ TRỢ
# =========================================================

def safe_text(value, default="Chưa cập nhật"):
    """Chuẩn hóa giá trị thiếu trước khi hiển thị."""
    if value is None or pd.isna(value):
        return default

    value = str(value).strip()
    return value if value else default


def truncate_text(value, max_words):
    """Giới hạn số từ của văn bản."""
    text = safe_text(value, "Chưa có thông tin.")
    words = text.split()

    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."

    return text


def get_company_by_id(company_id):
    """Lấy thông tin công ty theo ID."""
    result = companies_df[companies_df["id"] == company_id]
    return None if result.empty else result.iloc[0]


def get_company_id(company_name):
    """Lấy ID công ty theo tên đã chọn."""
    result = company_options[
        company_options["Company Name"] == company_name
    ]
    return None if result.empty else result.iloc[0]["id"]


def render_company_overview(company):
    """Hiển thị thông tin cơ bản của công ty."""
    fields = [
        ("Lĩnh vực", "Company industry"),
        ("Loại hình", "Company Type"),
        ("Quy mô", "Company size"),
        ("Quốc gia", "Country"),
        ("Ngày làm việc", "Working days"),
        ("Chính sách làm thêm", "Overtime Policy"),
    ]
    
    cols = st.columns(3)

    for index, (label, key) in enumerate(fields):
        with cols[index % 3]:
            st.caption(label)
            st.write(safe_text(company.get(key)))

    overview = company.get("Company overview")
    if pd.notna(overview) and str(overview).strip():
        with st.expander("Tổng quan công ty"):
            st.write(str(overview))

    skills = company.get("Our key skills")
    if pd.notna(skills) and str(skills).strip():
        with st.expander("Kỹ năng chính"):
            st.write(str(skills))

    why_love = company.get("Why you'll love working_here")
    if pd.notna(why_love) and str(why_love).strip():
        with st.expander("Vì sao nên làm việc tại đây"):
            st.write(str(why_love))

# =========================================================
# 6. Hàm chức năng RECOMMENDATION
# =========================================================    

def get_top_k_from_scores(scores: np.ndarray, query_index=None, top_k=5):
    """ Lấy top-k document có score cao nhất """
    
    scores = scores.copy()

    # Nếu query là một document trong dataset Loại chính query khỏi kết quả
    if query_index is not None:
        scores[query_index] = -1.0

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(int(idx), float(scores[idx]))for idx in top_indices]


def build_recommendation_result(indices, scores, companies_df):
    """Tạo DataFrame kết quả đề xuất."""
    
    result = companies_df.iloc[indices].copy()
    result["similarity"] = np.asarray(scores) * 100
    result.insert(0, "rank", range(1, len(result) + 1))

    return result.reset_index(drop=True)


def recommend_similar_company(company_id, companies_df, similarity_matrix, top_n=5):
    """Đề xuất các công ty tương tự một công ty đã chọn."""
    
    # Tìm index của công ty
    matching = np.flatnonzero(companies_df["id"].to_numpy() == company_id)

    if len(matching) == 0:
        return pd.DataFrame()

    company_index = matching[0]

    top_results = get_top_k_from_scores(
        similarity_matrix[company_index],
        query_index=company_index,
        top_k=top_n
    )
    
    indices = [index for index, _ in top_results]
    scores = [score for i, score in top_results]
    
    return build_recommendation_result(indices, scores, companies_df)


def recommend_by_content(user_input, vectorizer, tfidf_matrix, companies_df, top_n=5):
    """Đề xuất công ty dựa trên nội dung người dùng nhập."""

    if not isinstance(user_input, str) or not user_input.strip():
        return pd.DataFrame()

    # 1. Chuyển nội dung user thành TF-IDF vector
    user_vector = vectorizer.transform([user_input])

    # 2. Tính similarity giữa user và toàn bộ công ty
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # 3. Lấy Top-N
    top_results = get_top_k_from_scores(similarities, top_k=top_n)

    indices = [index for index, _ in top_results]
    scores = [score for i, score in top_results]

    return build_recommendation_result(indices, scores, companies_df)
  
  
def display_company_card(row):
    """Hiển thị thẻ công ty trong kết quả đề xuất."""

    company_name = safe_text(row.get("Company Name"))
    rank = row.get("rank")
    similarity = row.get("similarity")
    
    title = (f"{int(rank)}. {company_name}" if pd.notna(rank) else company_name)
        
    with st.container(border=True):
        st.subheader(title)

        if pd.notna(similarity):
            st.caption(f"Điểm tương đồng: {similarity:.1f}/100")

        columns = st.columns(4)
        
        fields = [
            ("Lĩnh vực", "Company industry"),
            ("Loại hình", "Company Type"),
            ("Quy mô", "Company size"),
            ("Quốc gia", "Country"),
        ]

        for col, (label, key) in zip(columns, fields):
            with col:
                st.caption(label)
                st.write(safe_text(row.get(key)))

        with st.expander("Xem chi tiết"): 
            st.markdown("**Tổng quan**")
            st.write(truncate_text(row.get("Company overview"), 60))

            st.markdown("**Kỹ năng chính**")
            st.write(truncate_text(row.get("Our key skills"), 30))

    st.markdown("---")


# =========================================================
# 7. Hàm chức năng REVIEW CLASSIFICATION
# =========================================================

# RATING_FIELDS = [
#     ("Overall Rating", "review_rating", "Đánh giá tổng thể"),
#     ("Salary & Benefits", "review_salary", "Lương và phúc lợi"),
#     ("Training & Learning", "review_training", "Đào tạo và phát triển"),
#     ("Management cares about me", "review_management", "Sự hỗ trợ từ quản lý"),
#     ("Culture & Work Environment", "review_culture", "Văn hóa và môi trường"),
#     ("Office & Workspace", "review_workspace", "Không gian làm việc"),
# ]

RATING_FIELDS = [
    ("Overall Rating", "review_rating", "Mức độ hài lòng chung"),
    ("Salary & Benefits", "review_salary", "Thu nhập, thưởng và chế độ đãi ngộ"),
    ("Training & Learning", "review_training", "Cơ hội phát triển kỹ năng"),
    ("Management cares about me", "review_management", "Hỗ trợ và ghi nhận từ cấp trên"),
    ("Culture & Work Environment", "review_culture", "Đồng nghiệp và hoạt động nội bộ"),
    ("Office & Workspace", "review_workspace", "Không gian và điều kiện làm việc"),
]


def rating_card(label, key, description, default=4):
    """Chọn điểm đánh giá từ 1 đến 5."""
    st.markdown(f"**{label}**")
    st.caption(description)

    if key not in st.session_state:
        st.session_state[key] = default

    cols = st.columns(5)

    for score, col in enumerate(cols, start=1):
        with col:
            st.button(
                str(score),
                key=f"{key}_btn_{score}",
                width=True,
                type=("primary" if st.session_state[key] == score else "secondary"),
                on_click=lambda k=key, s=score: st.session_state.update({k: s}),
            )

    st.caption(f"Điểm đã chọn: {st.session_state[key]}/5")
    return st.session_state[key]


def render_review_classification(company_name, company):
    """Thu thập review và thực hiện dự đoán."""
    st.subheader("Phân loại đánh giá")

    st.write("Nhập nội dung đánh giá")

    rating_values = {}

    # cols = st.columns(3, gap="large")

    # for index, (label, key, description) in enumerate(RATING_FIELDS):
    #     with cols[index % 3]:
    #         rating_values[key] = rating_card(label, key, description)

    cols = st.columns(3, gap="medium")

    for index, (label, key, description) in enumerate(RATING_FIELDS):
        with cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(description)

                rating_values[key] = st.radio(
                    "Điểm đánh giá",
                    options=[1, 2, 3, 4, 5],
                    horizontal=True,
                    index=3,
                    key=f"rating_{key}",
                    label_visibility="collapsed",
                )

    title = st.text_input(
        "Tiêu đề",
        placeholder="Ví dụ: Môi trường làm việc tốt",
        key="review_title",
    )

    liked = st.text_area(
        "Điều bạn thích",
        placeholder="Ví dụ: Đồng nghiệp thân thiện...",
        height=100,
        key="review_liked",
    )

    suggestions = st.text_area(
        "Điều bạn muốn cải thiện",
        placeholder="Ví dụ: Cải thiện chính sách nghỉ phép...",
        height=100,
        key="review_suggest",
    )

    if not st.button(
        "Đánh giá",
        type="primary",
        width="stretch",
        key="classify_review",
    ):
        return

    if not any(text.strip() for text in (title, liked, suggestions)):
        st.warning("Vui lòng nhập ít nhất một nội dung đánh giá.")
        return

    company_type = company.get("Company_Type")
    overtime_policy = company.get("Overtime_Policy")

    review_input = pd.DataFrame([{
        "Company Name": company_name,
        "Company Type": company_type,
        "Overtime Policy": overtime_policy,
        "Rating": rating_values["review_rating"],
        "Salary & benefits": rating_values["review_salary"],
        "Training & learning": rating_values["review_training"],
        "Management cares about me": rating_values["review_management"],
        "Culture & fun": rating_values["review_culture"],
        "Office & workspace": rating_values["review_workspace"],
        "Title": title,
        "What I liked": liked,
        "Suggestions for improvement": suggestions,
    }])

    try:
        prepared_review = prepare_data(review_input)

        detail_df, _ = predict_with_class0_threshold(
            prepared_review,
            clf_model,
            clf_threshold,
        )

        if detail_df.empty:
            st.error("Mô hình không trả về kết quả dự đoán.")
            return

        result = detail_df.iloc[0]

        is_recommend = int(result["is_recommend"])
        prob_class0 = float(result["prob_class0"])
        prob_class1 = float(result["prob_class1_recommend"])

    except (ValueError, KeyError, TypeError, AttributeError) as error:
        st.error(f"Không thể phân loại review: {error}")
        return

    st.divider()
    st.subheader("Kết quả dự đoán")

    if is_recommend == 1:
        st.success("RECOMMEND")
        st.write(f"Review có xu hướng giới thiệu {company_name}.")
    else:
        st.error("NOT RECOMMEND")
        st.write(f"Review có xu hướng không giới thiệu {company_name}.")

    # cols = st.columns(2)

    # with cols[0]:
    #     st.metric("Recommend", f"{prob_class1 * 100:.1f}%")
    #     st.progress(float(np.clip(prob_class1, 0, 1)))

    # with cols[1]:
    #     st.metric("Not Recommend", f"{prob_class0 * 100:.1f}%")
    #     st.progress(float(np.clip(prob_class0, 0, 1)))
        

def render_employee_insights(company_id):
    """Hiển thị thống kê đánh giá của công ty."""
    st.subheader("Tổng quan đánh giá")

    if "id" not in overview_reviews.columns:
        st.warning("Dữ liệu Overview_Reviews thiếu cột 'id'.")
        return

    review_stats = overview_reviews[overview_reviews["id"] == company_id]

    if review_stats.empty:
        st.info("Chưa có dữ liệu đánh giá cho công ty này.")
        return

    stats = review_stats.iloc[0]

    number_reviews = pd.to_numeric(stats.get("Number of reviews"), errors="coerce")
    
    if pd.notna(number_reviews) and number_reviews > 0:
        st.caption(f"Số lượng đánh giá: {int(number_reviews)}")
    else:
        st.caption("Số lượng đánh giá: Chưa có dữ liệu")

    overall_rating = pd.to_numeric(stats.get("Overall rating"), errors="coerce")

    if pd.isna(overall_rating) or overall_rating <= 0:
        st.info("Chưa có thông tin điểm đánh giá.")
        return

    rating_columns = [
        ("Lương & phúc lợi", "Salary & benefits"),
        ("Đào tạo & học tập", "Training & learning"),
        ("Quan tâm từ quản lý", "Management cares about me"),
        ("Văn hóa & giải trí", "Culture & fun"),
        ("Môi trường làm việc", "Office & workspace"),
    ]

    cols = st.columns(5)

    for col, (label, key) in zip(cols, rating_columns):
        value = pd.to_numeric(stats.get(key), errors="coerce")

        with col:
            if pd.notna(value) and value > 0:
                st.metric(label, f"{value:.1f}/5")
            else:
                st.metric(label, "N/A")

    # st.markdown("**Khả năng giới thiệu công ty**")

    recommend_value = stats.get("Recommend working here to a friend")

    if pd.isna(recommend_value):
        st.info("Chưa có dữ liệu.")
    else:
        # Xử lý giá trị dạng "83%" hoặc số
        if isinstance(recommend_value, str):
            recommend_value = recommend_value.strip().replace("%", "")

        recommend_value = pd.to_numeric(recommend_value, errors="coerce")

        if pd.isna(recommend_value):
            st.info("Dữ liệu tỷ lệ giới thiệu không hợp lệ.")
        else:
            # Chuyển tỷ lệ thập phân thành phần trăm
            if 0 < recommend_value <= 1:
                recommend_value *= 100

            recommend_value = float(np.clip(recommend_value, 0, 100))

            st.progress(recommend_value / 100)
            st.write(f"{recommend_value:.0f}% khuyến khích làm việc tại đây.")

# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("💼 IT Company")
    st.caption("Recommender & Insights")
    st.divider()

    menu_options = [
        "Trang chủ",
        "Company Recommendation",
        "Company Review",
        "Giới thiệu dự án",
    ]

    if "home_navigation" in st.session_state:
        st.session_state["page_menu"] = (
            st.session_state.pop("home_navigation")
        )

    if "page_menu" not in st.session_state:
        st.session_state["page_menu"] = "Trang chủ"

    menu = st.radio(
        "Điều hướng",
        menu_options,
        key="page_menu",
        label_visibility="visible",
    )
    
    st.markdown(
        '<div style="margin-top: 220px;"></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Thông tin nhóm")

    st.markdown(
        """
        **Bùi Thị Thư**  
        buithithu.ntt@gmail.com

        **Lê Thị Hà My**  
        lehamy.yds@gmail.com
        """
    )

# =========================================================
# 9. TRANG CHỦ
# =========================================================

BANNER_PATH  = r"channels4_banner.jpg"

if menu == "Trang chủ":

    st.title("IT COMPANY RECOMMENDER SYSTEM")
    
    st.markdown(
    """
    <p style="
        color: #64748B;
        font-size: 16px;
        margin-top: -10px;
        margin-bottom: 28px;
    ">
        Khám phá công ty công nghệ, tìm kiếm môi trường làm việc
        phù hợp và phân tích đánh giá từ nhân viên.
    </p>
    """,
    unsafe_allow_html=True,
    )
    
    # Hiển thị banner
    st.image(BANNER_PATH, width="stretch")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Phần giới thiệu
    st.subheader("Khám phá hệ thống")
    
    st.caption(
        "Lựa chọn một chức năng bên dưới để bắt đầu."
    )
    

    col1, col2 = st.columns(2, gap="large")

    # Thẻ Company Recommendation
    with col1:
        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    color: #64748B;
                    font-size: 13px;
                    font-weight: 600;
                    margin-bottom: 8px;
                ">
                    FEATURE 01
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Company Recommendation")

            st.write(
                "Tìm kiếm các công ty công nghệ theo kỹ năng, "
                "lĩnh vực và nhu cầu nghề nghiệp. "
                "Khám phá những công ty có nội dung tương đồng."
            )

            st.markdown(
                """
                <div style="
                    color: #64748B;
                    font-size: 13px;
                    margin-top: 16px;
                ">
                    TF-IDF · Cosine Similarity
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Khám phá công ty",
                key="home_recommend_button",
                type="primary",
                width="stretch",
            ):
                st.session_state["home_navigation"] = (
                    "Company Recommendation"
                )
                st.rerun()

    # Thẻ Company Review
    with col2:
        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    color: #64748B;
                    font-size: 13px;
                    font-weight: 600;
                    margin-bottom: 8px;
                ">
                    FEATURE 02
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Company Review")

            st.write(
                "Xem thông tin cơ bản về công ty, "
                "các chỉ số đánh giá từ nhân viên "
                "xu hướng của đánh giá (Recommend hoặc Not Recommend)."
            )

            st.markdown(
                """
                <div style="
                    color: #64748B;
                    font-size: 13px;
                    margin-top: 16px;
                ">
                    Logistic Regression · Employee Insights
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Xem đánh giá",
                key="home_review_button",
                type="primary",
                width="stretch",
            ):
                st.session_state["home_navigation"] = (
                    "Company Review"
                )
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Thông tin tổng quan
    # st.subheader("Tổng quan dữ liệu")

    # metric1, metric2, metric3 = st.columns(3, gap="medium")

    # with metric1:
    #     with st.container(border=True):
    #         st.caption("Công ty trong hệ thống")
    #         st.metric(
    #             label="Companies",
    #             value=f"{len(companies_df):,}",
    #         )

    # with metric2:
    #     with st.container(border=True):
    #         st.caption("Bản ghi đánh giá tổng quan")
    #         st.metric(
    #             label="Review Overview",
    #             value=f"{len(overview_reviews):,}",
    #         )

    # with metric3:
    #     with st.container(border=True):
    #         st.caption("Bản ghi đánh giá chi tiết")
    #         st.metric(
    #             label="Reviews",
    #             value=f"{len(reviews):,}",
    #         )

    st.markdown("<br>", unsafe_allow_html=True)

    # st.caption(
    #     "IT Company Recommender System | "
    #     "Data-driven insights for IT career exploration"
    # )

# =========================================================
# 10. COMPANY RECOMMENDATION
# =========================================================

elif menu == "Company Recommendation":
    st.title("Company Recommendation")
    st.caption("Tìm công ty phù hợp với nhu cầu của bạn.")
    st.divider()

    search_method = st.radio(
        "Phương thức tìm kiếm",
        ["Theo nhu cầu", "Theo tên công ty"],
        horizontal=True,
        key="recommend_search_method",
    )

    if search_method == "Theo nhu cầu":
        st.subheader("Tìm công ty theo nhu cầu")
        st.info(
            "💡 Bạn nên sử dụng các từ khóa tiếng Anh đơn giản, "
            "ví dụ: Python, SQL, Data Analyst, Fintech, "
            "good working environment..."
        )

        user_input = st.text_area(
            "Bạn đang tìm kiếm điều gì?",
            placeholder="Ví dụ: Data Analyst, Python, Power BI, ...",
            height=120,
            key="recommend_user_input",
        )

        top_n = st.slider(
            "Số lượng công ty đề xuất",
            min_value=1,
            max_value=10,
            value=5,
            key="recommend_text_top_n",
        )

        if st.button(
            "Tìm công ty phù hợp",
            type="primary",
            width="stretch",
            key="recommend_by_content_button",
        ):
            if not user_input.strip():
                st.warning("Vui lòng nhập nhu cầu tìm kiếm.")
            else:
                result = recommend_by_content(
                    user_input,
                    vectorizer,
                    tfidf_matrix,
                    companies_df,
                    top_n
                )

                if result.empty:
                    st.info("Không tìm thấy kết quả phù hợp.")
                else:
                    st.success(f"Tìm thấy {len(result)} công ty.")

                    for _, row in result.iterrows():
                        display_company_card(row)

    else:
        st.subheader("Tìm công ty tương tự")

        selected_company = st.selectbox(
            "Chọn công ty",
            options=company_options["Company Name"].tolist(),
            key="recommend_selected_company",
        )

        selected_id = get_company_id(selected_company)
        company = get_company_by_id(selected_id)

        if company is not None:
            with st.container(border=True):
                st.subheader(safe_text(company.get("Company Name")))
                render_company_overview(company)

        top_n = st.slider(
            "Số lượng công ty tương tự",
            min_value=1,
            max_value=10,
            value=5,
            key="recommend_company_top_n",
        )

        if st.button(
            "Tìm công ty tương tự",
            type="primary",
            width="stretch",
            key="recommend_by_company_button",
        ):
            result = recommend_similar_company(
                    selected_id,
                    companies_df,
                    similarity_matrix,
                    top_n
                )

            if result.empty:
                st.info("Không tìm thấy công ty tương tự.")
            else:
                st.success(
                    f"Các công ty tương tự với {selected_company}"
                )

                for _, row in result.iterrows():
                    display_company_card(row)
                    
# =========================================================
# 11. COMPANY REVIEW PAGE
# =========================================================

elif menu == "Company Review":
    st.title("Company Review & Classification")
    st.caption("Xem thông tin cơ bản về công ty, các chỉ số đánh giá từ nhân viên và thực hiện đánh giá công ty.")
    st.divider()

    selected_company = st.selectbox(
        "Chọn công ty",
        options=company_options["Company Name"].tolist(),
        key="classification_company",
    )

    selected_id = get_company_id(selected_company)
    company = get_company_by_id(selected_id)

    if company is None:
        st.warning("Không tìm thấy thông tin công ty.")
    else:
        st.subheader("Company Overview")
        render_company_overview(company)

        st.divider()
        render_employee_insights(selected_id)

        st.divider()
        render_review_classification(selected_company, company)
        
# =========================================================
# GIỚI THIỆU DỰ ÁN
# =========================================================

elif menu == "Giới thiệu dự án":
    st.title("Giới thiệu dự án")

    with st.container(border=True):
        st.subheader("Mục tiêu")
        st.write(
            "Xây dựng hệ thống hỗ trợ tìm kiếm, khám phá và đánh giá "
            "các công ty công nghệ dựa trên dữ liệu."
        )

    with st.container(border=True):
        st.subheader("1. Company Recommendation")
        st.write(
            "Sử dụng phương pháp Content-Based Recommendation "
            "dựa trên TF-IDF và Cosine Similarity để tìm các công ty có nội dung tương đồng."
        )
  
    with st.container(border=True):
        st.subheader("2. Review Classification")
        st.write(
            "Sử dụng Logistic Regression để dự đoán xu hướng "
            "Recommend hoặc Not Recommend từ nội dung và điểm đánh giá."
        )
        st.write(
            "Đây là mô hình tốt nhất sau khi thử nghiệm trên 4 mô hình: "
            "Logistic Regression, Linear SVM, KNN, Random Forest ở môi trường Scikit-Learn."                    
        )
        st.caption(
            f"Ngưỡng xác suất Class 0 đang sử dụng: {clf_threshold:.2f}"
        )
        st.caption(
            "Mô hình Logistic Regression * Precision: 0.7938 * Recall: 0.8213 * F1-Score: 0.7681 *"
        )

    st.info(
        "Kết quả phân loại là dự đoán của mô hình"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        <div>IT Company Recommender System | Data-driven insights for IT career exploration</div>
        <div>Dataset: <strong>{len(companies_df)} companies</strong></div>
    </div>
    """,
    unsafe_allow_html=True
)
