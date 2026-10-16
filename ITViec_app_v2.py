import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# SỬA 5: file này có thể nằm ở helps/ hoặc ở helps/ITViec/. Tự tìm thư mục chứa ITviec_models1 + preprocessing.py,
# rồi chdir vào đó (preprocessing.py đọc "data/english-vnmese.txt" theo đường dẫn tương đối).
_HERE = Path(__file__).resolve().parent
APP_DIR = _HERE if (_HERE / "ITviec_models1").exists() else _HERE / "ITViec"
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from preprocessing import prepare_data, predict_with_class0_threshold

# =========================================================
# 1. CẤU HÌNH TRANG & NẠP CSS TẬP TRUNG
# =========================================================

st.set_page_config(
    page_title="ITViec Recommender & Sentiment Insights",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = APP_DIR
# SỬA 3: ưu tiên style.css của nhóm;
# Bản cũ chỉ nạp style.css và im lặng bỏ qua khi thiếu, nên mọi class (itv-badge, spec-grid, ...) mất định dạng.
CSS_FILE = BASE_DIR / "style.css"
if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Không tìm thấy `style.css` cạnh file app nên giao diện sẽ không có định dạng.")


# =========================================================
# 2. LOAD MODEL & DỮ LIỆU TỐI ƯU HÓA BỘ NHỚ
# =========================================================

MODEL_DIR = BASE_DIR / "ITviec_models1"
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def load_models():
    rec_path = MODEL_DIR / "company_recommender_bundle.joblib"
    clf_path = MODEL_DIR / "best_logistic_model.pkl"
    if not rec_path.exists() or not clf_path.exists():
        raise FileNotFoundError("Không tìm thấy tệp mô hình trong thư mục ITviec_models1.")
    return joblib.load(rec_path), joblib.load(clf_path)

@st.cache_data(show_spinner="Đang đọc cơ sở dữ liệu ITViec...")
def load_datasets():
    ov_rev = DATA_DIR / "Overview_Reviews.xlsx"
    rev = DATA_DIR / "Reviews.xlsx"
    if not (ov_rev.exists() and rev.exists()):
        raise FileNotFoundError("Không tìm thấy các tệp dữ liệu Excel trong thư mục data.")
    
    # Đọc tổng quan đánh giá công ty
    ov_rev_df = pd.read_excel(ov_rev)
    # Tối ưu: Chỉ đếm số lượng review để thống kê, không nạp toàn bộ bảng review nặng vào RAM
    total_reviews = len(pd.read_excel(rev, usecols=[0]))
    return ov_rev_df, total_reviews

try:
    rec_model, clf_bundle = load_models()
    overview_reviews, total_reviews_count = load_datasets()
except Exception as error:
    st.error(f"❌ Không thể tải hệ thống: {error}")
    st.stop()

# Thành phần Recommendation Model
vectorizer = rec_model["vectorizer"]
tfidf_matrix = rec_model["tfidf_matrix"]
similarity_matrix = rec_model["similarity_matrix"]
companies_df = rec_model["companies_df"]

if (
    len(companies_df) != tfidf_matrix.shape[0] 
    or similarity_matrix.shape != (len(companies_df), len(companies_df))
):
    st.error("❌ Recommendation Model không đồng bộ kích thước.")
    st.stop()

# Thành phần Classification Model
clf_model = clf_bundle["pipeline"]
clf_threshold = float(clf_bundle["best_threshold_class0"])

# Danh sách công ty và tra cứu nhanh O(1)
company_options = (
    companies_df[["id", "Company Name"]]
    .sort_values("Company Name", na_position="last")
    .reset_index(drop=True)
)
company_names_list = company_options["Company Name"].tolist()
name_to_id = dict(zip(company_options["Company Name"], company_options["id"]))
id_to_company = companies_df.set_index("id").to_dict(orient="index")
id_to_overview_reviews = (
    overview_reviews.set_index("id").to_dict(orient="index") 
    if "id" in overview_reviews.columns else {}
)

# =========================================================
# 3. HÀM TIỆN ÍCH DỮ LIỆU & UI COMPONENTS DÙNG CHUNG
# =========================================================

def safe_text(value, default="Chưa cập nhật") -> str:
    """Chuẩn hóa giá trị thiếu trước khi hiển thị."""
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def parse_percent_value(value) -> float:
    """Chuẩn hóa tỷ lệ phần trăm an toàn sang khoảng [0, 100]."""
    if pd.isna(value):
        return 0.0
    val_clean = str(value).replace("%", "").strip()
    num = pd.to_numeric(val_clean, errors="coerce")
    if pd.isna(num):
        return 0.0
    if 0 < num <= 1:
        num *= 100
    return float(np.clip(num, 0, 100))


def render_skills_badges(skills_val) -> str:
    """Bóc tách chuỗi kỹ năng và sinh HTML badges đồng bộ."""
    if pd.isna(skills_val) or not str(skills_val).strip():
        return "Chưa cập nhật danh sách kỹ năng."
    tokens = [s.strip() for s in str(skills_val).replace(";", ",").split(",") if s.strip()]
    if not tokens:
        return str(skills_val)
    return " ".join([f'<span class="itv-badge itv-badge-blue" style="margin: 2px 3px;">{t}</span>' for t in tokens])


def render_spec_grid(company_data) -> str:
    """Component sinh HTML lưới 6 cột thông số công ty (dùng chung cho mọi nơi)."""
    if isinstance(company_data, pd.Series):
        c_dict = company_data.to_dict()
    elif isinstance(company_data, dict):
        c_dict = company_data
    else:
        c_dict = {}

    industry = safe_text(c_dict.get("Company industry"))
    comp_type = safe_text(c_dict.get("Company Type", c_dict.get("Company_Type")))
    comp_size = safe_text(c_dict.get("Company size"))
    country = safe_text(c_dict.get("Country"))
    working_days = safe_text(c_dict.get("Working days"))
    ot_policy = safe_text(c_dict.get("Overtime Policy", c_dict.get("Overtime_Policy")))

    return f"""
    <div class="spec-grid">
        <div class="spec-item">
            <div class="spec-label">🏢 Lĩnh vực</div>
            <div class="spec-val">{industry}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">📁 Loại hình</div>
            <div class="spec-val"><span class="itv-badge itv-badge-blue">{comp_type}</span></div>
        </div>
        <div class="spec-item">
            <div class="spec-label">👥 Quy mô</div>
            <div class="spec-val">{comp_size}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">📍 Quốc gia</div>
            <div class="spec-val">{country}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">📅 Ngày làm việc</div>
            <div class="spec-val">{working_days}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">⚡ Chính sách OT</div>
            <div class="spec-val"><span class="itv-badge itv-badge-green">{ot_policy}</span></div>
        </div>
    </div>
    """


def render_company_overview_card(company):
    """Hiển thị thẻ tổng quan công ty."""
    st.markdown(render_spec_grid(company), unsafe_allow_html=True)
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)    

    overview = company.get("Company overview")
    if pd.notna(overview) and str(overview).strip():
        with st.expander("📖 Tổng quan về công ty (Company overview)", expanded=False):
            st.markdown(str(overview).replace('\n', '<br>'), unsafe_allow_html=True)

    skills = company.get("Our key skills")
    if pd.notna(skills) and str(skills).strip():
        with st.expander("🛠️ Kỹ năng & Công nghệ chính (Our key skills)", expanded=True):
            st.markdown(render_skills_badges(skills), unsafe_allow_html=True)

    why_love = company.get("Why you'll love working here", company.get("Why you'll love working_here"))
    if pd.notna(why_love) and str(why_love).strip():
        with st.expander("❤️ Điểm thu hút nhân tài (Why you'll love working here)", expanded=False):
            st.markdown(str(why_love).replace('\n', '<br>'), unsafe_allow_html=True)


def set_recommend_query(text):
    """SỬA 2: callback cho nút gợi ý nhanh; chạy trước rerun nên được phép đặt giá trị cho ô nhập."""
    st.session_state["recommend_user_input"] = text


# =========================================================
# 4. THUẬT TOÁN ĐỀ XUẤT (RECOMMENDATION)
# =========================================================
def get_top_k_indices_scores(scores: np.ndarray, query_index=None, top_k=5):
    """Lấy danh sách index và điểm số tương đồng Top-K."""
    s = np.asarray(scores).ravel().copy()
    if query_index is not None:
        if not 0 <= query_index < len(s):
            raise IndexError("query_index nằm ngoài phạm vi dataset.")
        s[query_index] = -np.inf
        max_k = len(s) - 1
    else:
        max_k = len(s)
    top_k = min(top_k, max_k)
    if top_k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    top_indices = np.argsort(s)[::-1][:top_k]
    return top_indices, s[top_indices]


def build_recommendation_result(indices, scores, companies_df):
    """Đóng gói kết quả đề xuất thành DataFrame."""
    result = companies_df.iloc[indices].copy()
    result["similarity"] = np.asarray(scores) * 100
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def recommend_similar_company(company_id, companies_df, similarity_matrix, top_n=5):
    """Đề xuất các công ty tương tự công ty được chọn."""
    matching = np.flatnonzero(companies_df["id"].to_numpy() == company_id)
    if len(matching) == 0:
        return pd.DataFrame()
    company_index = matching[0]
    scores = np.asarray(similarity_matrix[company_index]).ravel()
    indices, scores = get_top_k_indices_scores(scores, query_index=company_index, top_k=top_n)
    return build_recommendation_result(indices, scores, companies_df)


def recommend_by_content(user_input, vectorizer, tfidf_matrix, companies_df, top_n=5):
    """Đề xuất công ty dựa trên nội dung tìm kiếm người dùng nhập."""
    if not isinstance(user_input, str) or not user_input.strip():
        return pd.DataFrame()
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    indices, scores = get_top_k_indices_scores(similarities, top_k=top_n)
    return build_recommendation_result(indices, scores, companies_df)


def display_company_card(row):
    """Hiển thị thẻ công ty kết quả đề xuất với phân cấp màu điểm số."""
    company_name = safe_text(row.get("Company Name"))
    rank = int(row.get("rank", 1))
    similarity = float(row.get("similarity", 0.0))

    rank_badge = "🥇 Hạng 1" if rank == 1 else ("🥈 Hạng 2" if rank == 2 else ("🥉 Hạng 3" if rank == 3 else f"#{rank}"))
    score_chip_class = "score-chip-high" if similarity >= 85.0 else "score-chip-mid"

    st.markdown(f"""
    <div class="company-result-card">
        <div class="company-header-row">
            <span class="itv-badge itv-badge-blue" style="font-weight: 700; flex-shrink: 0;">{rank_badge}</span>
            <div class="company-name-title" style="flex: 1;">{company_name}</div>
            <div class="score-chip {score_chip_class}" style="flex-shrink: 0;">
                <span style="font-size: 0.78rem; opacity: 0.85;">🎯 Độ tương đồng: </span>
                <span>{similarity:.1f}%</span>
            </div>
        </div>
        {render_spec_grid(row)}
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🔍 Xem thông tin chi tiết về {company_name}"):
        tab_ov, tab_sk, tab_why = st.tabs(["Tổng quan", "Kỹ năng chính", "Môi trường & Đãi ngộ"])
        with tab_ov:
            st.write(safe_text(row.get("Company overview"), "Chưa có mô tả tổng quan."))
        with tab_sk:
            st.markdown(render_skills_badges(row.get("Our key skills")), unsafe_allow_html=True)
        with tab_why:
            why_text = row.get("Why you'll love working here", row.get("Why you'll love working_here"))
            st.write(safe_text(why_text, "Chưa cập nhật lý do nên làm việc."))


# =========================================================
# 5. HEADER DASHBOARD CÔNG TY & PHÂN TÍCH REVIEW
# =========================================================

ASPECTS_INFO = [
    ("Lương & Phúc lợi", "Salary & benefits"),
    ("Đào tạo & Phát triển", "Training & learning"),
    ("Quản lý quan tâm", "Management cares about me"),
    ("Văn hóa & Giải trí", "Culture & fun"),
    ("Môi trường làm việc", "Office & workspace"),
]

def render_company_dashboard_header(selected_company, company, company_id):
    """Hiển thị giao diện Executive Dashboard chia đôi (Split Layout) cho trang Company Review."""
    if company is None:
        st.warning("⚠️ Không tìm thấy thông tin công ty.")
        return

    industry = safe_text(company.get("Company industry"))
    comp_type = safe_text(company.get("Company Type", company.get("Company_Type")))
    comp_size = safe_text(company.get("Company size"))
    country = safe_text(company.get("Country"))
    working_days = safe_text(company.get("Working days"))
    ot_policy = safe_text(company.get("Overtime Policy", company.get("Overtime_Policy")))

    stats = id_to_overview_reviews.get(company_id, {})
    if stats:
        number_reviews = pd.to_numeric(stats.get("Number of reviews"), errors="coerce")
        overall_rating = pd.to_numeric(stats.get("Overall rating"), errors="coerce")
        rec_num = parse_percent_value(stats.get("Recommend working here to a friend"))
    else:
        number_reviews, overall_rating, rec_num = None, None, 0.0

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">📋 HỒ SƠ CÔNG TY</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-kv-grid">
            <div class="kv-item">
                <span class="kv-label">🏢 Lĩnh vực</span>
                <span class="kv-val">{industry}</span>
            </div>
            <div class="kv-item">
                <span class="kv-label">📁 Loại hình</span>
                <span class="kv-val"><span class="itv-badge itv-badge-blue">{comp_type}</span></span>
            </div>
            <div class="kv-item">
                <span class="kv-label">👥 Quy mô</span>
                <span class="kv-val">{comp_size}</span>
            </div>
            <div class="kv-item">
                <span class="kv-label">📍 Quốc gia</span>
                <span class="kv-val">{country}</span>
            </div>
            <div class="kv-item">
                <span class="kv-label">📅 Ngày làm việc</span>
                <span class="kv-val">{working_days}</span>
            </div>
            <div class="kv-item">
                <span class="kv-label">⚡ Chính sách OT</span>
                <span class="kv-val"><span class="itv-badge itv-badge-green">{ot_policy}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        overview = company.get("Company overview")
        if pd.notna(overview) and str(overview).strip():
            with st.expander("📖 Xem mô tả tổng quan công ty"):
                st.markdown(str(overview).replace('\n', '<br>'), unsafe_allow_html=True)

        skills = company.get("Our key skills")
        if pd.notna(skills) and str(skills).strip():
            with st.expander("🛠️ Kỹ năng & Công nghệ chính"):
                st.markdown(render_skills_badges(skills), unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">📊 ĐÁNH GIÁ TỪ CỘNG ĐỒNG</div>', unsafe_allow_html=True)
        has_review_data = bool(stats) and pd.notna(overall_rating) and overall_rating > 0

        if not has_review_data:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 10px; padding: 35px 20px; text-align: center; margin-top: 10px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">💬</div>
                <div style="font-weight: 600; color: #334155; font-size: 0.95rem; margin-bottom: 4px;">Chưa có dữ liệu đánh giá</div>
                <div style="color: #64748B; font-size: 0.85rem;">Công ty này chưa nhận được review nào từ cộng đồng. Hãy kéo xuống dưới để chia sẻ góc nhìn của bạn nhé!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            m1, m2, m3 = st.columns([1, 1, 1])
            with m1:
                st.metric("Điểm Hài Lòng", f"{overall_rating:.1f} ⭐")
            with m2:
                st.metric("Lượt Đánh Giá", f"{int(number_reviews):,}" if pd.notna(number_reviews) else "0")
            with m3:
                st.metric("Khuyên Làm", f"{rec_num:.0f}%" if rec_num > 0 else "N/A")

            st.markdown("<hr style='margin: 10px 0 14px 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>ĐIỂM TRUNG BÌNH THEO 5 KHÍA CẠNH:</div>", unsafe_allow_html=True)

            for label, key in ASPECTS_INFO:
                val = pd.to_numeric(stats.get(key), errors="coerce")
                if pd.notna(val) and val > 0:
                    pct = (val / 5.0) * 100
                    score_str = f"{val:.1f}"
                    color_cls = "aspect-bar-high" if val >= 4.0 else ("aspect-bar-med" if val >= 3.0 else "aspect-bar-low")
                else:
                    pct = 0
                    score_str = "N/A"
                    color_cls = "aspect-bar-med"

                st.markdown(f"""
                <div class="aspect-row">
                    <div class="aspect-label" title="{label}">{label}</div>
                    <div class="aspect-bar-bg">
                        <div class="aspect-bar-fill {color_cls}" style="width: {pct}%;"></div>
                    </div>
                    <div class="aspect-score">{score_str}</div>
                </div>
                """, unsafe_allow_html=True)


def render_review_classification(company_name, company):
    """Thu thập đánh giá của người dùng và dự đoán xu hướng bằng AI."""
    st.markdown("### ✍️ Đánh Giá Công Ty & Dự Đoán Xu Hướng")
    st.caption("Nhấp chọn số ngôi sao tương ứng cho từng khía cạnh và chia sẻ cảm nghĩ. Mô hình Logistic Regression sẽ phân tích nội dung để dự đoán xu hướng.")

    rating_values = {}


    c_row1_l, c_row1_r = st.columns([1, 1], gap="medium")
        
    with c_row1_l:
        with st.container(border=True, key="review_rating_panel"):
            st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>⭐ Đánh giá & Xếp hạng</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 12px 0 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
            
            lbl, star_icon = st.columns([1, 1.2], vertical_alignment="center")
            with lbl:
                st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>🎯 Đánh giá tổng quan </div>", unsafe_allow_html=True)
            # st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
            
            with star_icon:
                try:
                    with st.container(key="overall_star_container"):
                        fb_val = st.feedback("stars", key="fb_overall")
                    rating_values["review_rating"] = (fb_val + 1) if fb_val is not None else 4
                except Exception:
                    rating_values["review_rating"] = st.select_slider(
                        "Overall", options=[1, 2, 3, 4, 5], value=4,
                        format_func=lambda x: "★" * x + "☆" * (5 - x),
                        key="slider_overall", label_visibility="collapsed"
                    )
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>🎯 Chi tiết 5 khía cạnh</div>", unsafe_allow_html=True)
            aspect_fields = [
                ("Lương & Chế độ đãi ngộ", "review_salary"),
                ("Cơ hội đào tạo & Phát triển", "review_training"),
                ("Sự quan tâm từ cấp quản lý", "review_management"),
                ("Văn hóa & Đồng nghiệp", "review_culture"),
                ("Không gian làm việc", "review_workspace"),
            ]
            
            for label, key in aspect_fields:
                c_lbl, c_star_icon = st.columns([1, 1.2], vertical_alignment="center")
                with c_lbl:
                    st.markdown(f"<div style='font-size: 0.83rem; font-weight: 600; color: #334155;'>{label}</div>", unsafe_allow_html=True)
                with c_star_icon:
                    try:
                        val = st.feedback("stars", key=f"fb_{key}")
                        rating_values[key] = (val + 1) if val is not None else 4
                    except Exception:
                        rating_values[key] = st.select_slider(
                            label=label,
                            options=[1, 2, 3, 4, 5],
                            value=4,
                            format_func=lambda x: f"{'★' * x}{'☆' * (5 - x)} ({x}/5)",
                            key=f"star_fallback_{key}",
                            label_visibility="collapsed"
                        )
                # st.markdown("<div style='margin-bottom: 4px;'></div>", unsafe_allow_html=True)
                
            

    with c_row1_r:
        with st.container(border=True, key="review_content_panel"):
            st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>📝 Nội dung nhận xét</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 12px 0 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>📌 Tiêu đề đánh giá </div>", unsafe_allow_html=True)
            title = st.text_input(
                "Tiêu đề đánh giá",
                placeholder="Ví dụ: Môi trường làm việc cởi mở, chế độ đãi ngộ tốt",
                key="review_title",
                label_visibility="collapsed"
            )
            
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>👍 Điều bạn thích nhất tại công ty</div>", unsafe_allow_html=True)
            liked = st.text_area(
                "Điều bạn thích nhất tại công ty",
                placeholder="Ví dụ: Đồng nghiệp nhiệt tình, sếp tâm lý, dự án công nghệ hay...",
                height=75,
                key="review_liked",
                label_visibility="collapsed"
            )

            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 12px;'>💡 Đề xuất cải thiện</div>", unsafe_allow_html=True)         
            suggestions = st.text_area(
                "Đề xuất cải thiện (Suggestions for improvement)",
                placeholder="Ví dụ: Cần cân bằng chính sách OT, thời gian làm việc...",
                height=75,
                key="review_suggest",
                label_visibility="collapsed"
            )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    submit_btn = st.button(
        "🚀 Phân Tích & Dự Đoán",
        type="primary",
        width="stretch",
        key="btn_run_classification"
    )

    if not submit_btn:
        return

    if not any(t.strip() for t in (title, liked, suggestions)):
        st.warning("⚠️ Vui lòng nhập tiêu đề hoặc ít nhất một nội dung nhận xét để mô hình phân tích ngôn ngữ tự nhiên.")
        return

    company_type = company.get("Company Type", company.get("Company_Type", "IT Product"))
    overtime_policy = company.get("Overtime Policy", company.get("Overtime_Policy", "No OT"))
   
    # Tinh gọn: Truyền trực tiếp văn bản vào DataFrame để prepare_data() xử lý đúng 1 lần (Tránh double cleaning)
    review_input = pd.DataFrame([{
        "Company Name": company_name,
        # SỬA 1: mô hình dùng 2 cột này làm đặc trưng (predict_with_class0_threshold cũng groupby theo chúng);
        # bản cũ để comment nên luôn lỗi KeyError ['Company Type', 'Overtime Policy'] not in index
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

    with st.spinner("Đang tiền xử lý ngôn ngữ tự nhiên (NLP) và suy luận mô hình Logistic Regression..."):
        try:
            prepared_review = prepare_data(review_input)
            detail_df, _ = predict_with_class0_threshold(
                prepared_review,
                clf_model,
                clf_threshold,
            )

            if detail_df.empty:
                st.error("❌ Mô hình không trả về kết quả dự đoán.")
                return

            result = detail_df.iloc[0]
            is_recommend = int(result["is_recommend"])
            prob_class0 = float(result["prob_class0"])
            prob_class1 = float(result["prob_class1_recommend"])

        except Exception as error:
            st.error(f"❌ Quá trình phân loại gặp lỗi: {error}")
            return

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    # st.markdown("### 🎯 Kết Quả Phân Tích & Dự Đoán Trí Tuệ Nhân Tạo")

    if is_recommend == 1:
        st.markdown(f"""
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 20px; display: flex; align-items: flex-start; gap: 16px; margin-bottom: 20px;">
            <div style="font-size: 2.2rem; line-height: 1;">🎉</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #065F46; margin-bottom: 4px;">
                    KẾT QUẢ: RECOMMEND (KHUYẾN NGHỊ GIA NHẬP)
                </div>
                <div style="font-size: 0.92rem; color: #047857; line-height: 1.5;">
                    Mô hình đánh giá đây là phản hồi <strong>tích cực</strong>. Nội dung nhận xét có xu hướng khuyến khích bạn bè và đồng nghiệp ứng tuyển vào <strong>{company_name}</strong>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 20px; display: flex; align-items: flex-start; gap: 16px; margin-bottom: 20px;">
            <div style="font-size: 2.2rem; line-height: 1;">⚠️</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #9F1239; margin-bottom: 4px;">
                    KẾT QUẢ: NOT RECOMMEND (CẦN CÂN NHẮC)
                </div>
                <div style="font-size: 0.92rem; color: #BE123C; line-height: 1.5;">
                    Mô hình nhận diện nhiều điểm quan ngại hoặc chưa thỏa đáng trong đánh giá. Người đánh giá có xu hướng thận trọng hoặc không khuyến khích gia nhập <strong>{company_name}</strong>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3, gap="medium")
    with col_p1:
        st.metric(
            label="Xác suất Khuyến khích (Class 1)",
            value=f"{prob_class1 * 100:.1f}%",
            # SỬA 6: quyết định thật dựa trên ngưỡng của Class 0, không phải mốc 0,5 của Class 1
            delta=f"Khuyến nghị nếu Class 0 < {clf_threshold * 100:.0f}%",
            delta_color="off",
        )
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-fill-c1" style="width: {float(np.clip(prob_class1, 0, 1)) * 100:.1f}%;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.metric(
            label="Xác suất Cần Cân Nhắc (Class 0)",
            value=f"{prob_class0 * 100:.1f}%",
            delta=f"Ngưỡng kích hoạt: {clf_threshold * 100:.0f}%"
        )
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-fill-c0" style="width: {float(np.clip(prob_class0, 0, 1)) * 100:.1f}%;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        with st.container(border=True):
            st.caption("Thuật toán phân loại")
            st.markdown("**Logistic Regression Pipeline**")
            st.caption(f"Ngưỡng tối ưu Class 0: `{clf_threshold:.2f}` (tối ưu hóa mất cân bằng nhãn)")


# =========================================================
# 6. THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# =========================================================

PAGES = [
    "🏠 Trang chủ",
    "🎯 Tìm công ty tương đồng",
    "📊 Dự đoán đánh giá",
    "ℹ️ Giới thiệu dự án",
]

with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <div style="
            background: linear-gradient(135deg, #123B5D 0%, #167DAD 100%);
            color: white;
            font-size: 1.5rem;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(18, 59, 93, 0.25);
        ">💼</div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #0F172A; line-height: 1.1;">ITViec Intelligence</div>
            <div style="font-size: 0.78rem; color: #475569; font-weight: 600;">Recommender & Insights</div>
        </div>
    </div>
    <div style="margin-bottom: 16px;">
        <span class="itv-badge itv-badge-green">● Hệ thống hoạt động</span>
    </div>
    """, unsafe_allow_html=True)
    
    # --- BỔ SUNG KHÓA LIÊN KẾT TRUNG GIAN ---
    if "nav_target" in st.session_state:
        st.session_state["current_page"] = st.session_state.pop("nav_target")

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = PAGES[0]

    selected_menu = st.radio(
        "Chức năng hệ thống",
        PAGES,
        key="current_page",
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top: 45px;'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #DCE6ED; border-radius: 14px; padding: 14px; box-shadow: 0 2px 6px rgba(18, 59, 93, 0.03);">
        <div style="font-weight: 700; color: #0F172A; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
            👥 Nhóm Tác Giả
        </div>
        <div style="margin-bottom: 10px;">
            <div style="font-weight: 700; color: #1E293B; font-size: 0.92rem;">Bùi Thị Thư</div>
            <a href="mailto:buithithu.ntt@gmail.com" style="color: #167DAD; font-size: 0.82rem; text-decoration: none;">buithithu.ntt@gmail.com</a>
        </div>
        <div>
            <div style="font-weight: 700; color: #1E293B; font-size: 0.92rem;">Lê Thị Hà My</div>
            <a href="mailto:lehamy.yds@gmail.com" style="color: #167DAD; font-size: 0.82rem; text-decoration: none;">lehamy.yds@gmail.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 7. TRANG 1: TRANG CHỦ (HOMEPAGE)
# =========================================================

if selected_menu == "🏠 Trang chủ":
    with st.container(key="home_hero"):
        st.markdown("""
        <div class="home-hero-content">
            <div class="home-hero-eyebrow">✦ &nbsp; ĐỒ ÁN · DL07 · DỮ LIỆU ITViec</div>
            <h1 class="home-hero-title">Tìm đúng công ty<br><span>Hiểu đúng môi trường</span></h1>
            <p class="home-hero-description">
                Khám phá các công ty phù hợp và tìm hiểu trải nghiệm của nhân viên qua phân tích dữ liệu chuyên sâu.
            </p>
        </div>
        """, unsafe_allow_html=True)

        cta_left, cta_right = st.columns(2, gap="medium")
        with cta_left:
            if st.button("Gợi ý công ty →", key="home_hero_btn_rec", type="primary", width="stretch"):
                st.session_state["nav_target"] = "🎯 Tìm công ty tương đồng"
                st.rerun()
        with cta_right:
            if st.button("Phân tích đánh giá →", key="home_hero_btn_review", width="stretch"):
                st.session_state["nav_target"] = "📊 Dự đoán đánh giá"
                st.rerun()

        st.markdown(f"""
        <div class="home-hero-stats">
            <div class="home-hero-stat">
                <div class="home-hero-stat-value">{len(companies_df):,}</div>
                <div class="home-hero-stat-label">Công ty</div>
            </div>
            <div class="home-hero-stat">
                <div class="home-hero-stat-value">{total_reviews_count:,}</div>
                <div class="home-hero-stat-label">Đánh giá từ nhân viên</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Khám phá các tính năng chính")
    col_f1, col_f2 = st.columns(2, gap="medium")


    with col_f1:
        with st.container(border=True, key="feature_company_recommendation"):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="itv-badge itv-badge-blue">TÍNH NĂNG 01</span>
                <span style="font-size: 1.5rem;">🎯</span>
            </div>
            <h3 style="margin-top: 0; margin-bottom: 4px;">Company Recommendation</h3>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.35; min-height: 0; margin-top: 0; margin-bottom: 10px;">
                Tìm kiếm công ty phù hợp theo công nghệ (Python, React, DevOps...) hoặc tìm công ty tương đồng với nơi bạn yêu thích.
            </p>
            <div style="margin-bottom: 10px;">
                <span class="itv-badge itv-badge-blue">TF-IDF</span>
                <span class="itv-badge itv-badge-blue">Cosine Similarity</span>
                <span class="itv-badge itv-badge-blue">Top-K Match</span>
            </div>
            """, unsafe_allow_html=True)


            # if st.button("Khám phá gợi ý công ty →", key="home_btn_rec", type="primary", width="content"):
            #     st.session_state["nav_target"] = "🎯 Tìm công ty tương đồng"
            #     st.rerun()

    with col_f2:
        with st.container(border=True, key="feature_company_review"):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="itv-badge itv-badge-green">TÍNH NĂNG 02</span>
                <span style="font-size: 1.5rem;">📊</span>
            </div>
            <h3 style="margin-top: 0; margin-bottom: 4px;">Company Review & Sentiment</h3>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.35; min-height: 0; margin-top: 0; margin-bottom: 10px;">
                Tra cứu chỉ số hài lòng thực tế. Ứng dụng mô hình Logistic Regression để đánh giá xu hướng (Recommend / Not Recommend).
            </p>
            <div style="margin-bottom: 10px;">
                <span class="itv-badge itv-badge-green">Logistic Regression</span>
                <span class="itv-badge itv-badge-green">NLP Text Classifier</span>
            </div>
            """, unsafe_allow_html=True)

            # if st.button("Xem đánh giá & Thử nghiệm →", key="home_btn_rev", type="primary", width="content"):
            #     st.session_state["nav_target"] = "📊 Dự đoán đánh giá"
            #     st.rerun()


# =========================================================
# 8. TRANG 2: TÌM CÔNG TY TƯƠNG ĐỒNG (RECOMMENDATION)
# =========================================================

elif selected_menu == "🎯 Tìm công ty tương đồng":
    st.markdown("""
    <div class="itv-hero-badge">🎯 Company Recommendation</div>
    <h1 style="margin-bottom: 6px;">Gợi Ý Công Ty Tương Đồng</h1>
    """, unsafe_allow_html=True)

    col_label, col_mode, col_space = st.columns([1.5, 2.4, 2], vertical_alignment="center")
    with col_label:
        st.markdown("<div style='font-size: 14px; font-weight: 600; color: #334155;'>Phương thức tìm kiếm</div>", unsafe_allow_html=True)

    with col_mode:
        search_mode = st.segmented_control(
            "Phương thức tìm kiếm",
            options=["✨ Theo kỹ năng", "🏢 Theo tên công ty"],
            default="✨ Theo kỹ năng",
            selection_mode="single",
            label_visibility="collapsed",
            key="company_search_mode",
            width="stretch",
        )

    if search_mode == "✨ Theo kỹ năng":
        with st.container(border=True, key="recommend_search_panel"):
            col_input, col_num, col_btn = st.columns([4, 0.5, 1], vertical_alignment="bottom")
            
            with col_input:
                user_input = st.text_input(
                    "Mô tả kỹ năng hoặc công việc mong muốn",
                    placeholder="Ví dụ: Python, Data Analyst, SQL...",
                    key="recommend_user_input",
                )
            
            with col_num:
                top_n = st.selectbox("Số lượng", options=[5, 10, 15], index=0, key="recommend_text_top_n")
            
            with col_btn:
                search_btn = st.button("🔍 Tìm Kiếm", type="primary", width="stretch", key="btn_run_text_rec")

            # Gợi ý nhanh dạng Chip Pill nhỏ gọn
            st.markdown("<div style='display: flex; align-items: center; gap: 8px; margin-top: 6px;'><span style='font-size: 0.8rem; color: #64748B; font-weight: 600;'>💡 Gợi ý nhanh:</span></div>", unsafe_allow_html=True)
            with st.container(key="suggestion_chips"):
                # SỬA 2: gán session_state sau khi text_input cùng key đã tạo gây StreamlitWidgetAlreadyInstantiatedError.
                # Dùng on_click (callback chạy TRƯỚC khi script rerun) và không cần st.rerun().
                chips = [("Python", "chip_python"), ("Data Analyst", "chip_da"),
                         ("ReactJS", "chip_react"), ("Golang", "chip_golang")]
                chip_cols = st.columns([0.8, 1.1, 0.8, 0.8, 4])
                for col, (label, key) in zip(chip_cols, chips):
                    with col:
                        st.button(label, key=key, on_click=set_recommend_query, args=(label,))

        if search_btn or user_input.strip():
            if not user_input.strip():
                st.warning("⚠️ Vui lòng nhập từ khóa kỹ năng hoặc nhu cầu để hệ thống tìm kiếm.")
            else:
                with st.spinner("Đang phân tích kỹ năng và tìm kiếm công ty phù hợp..."):
                    result = recommend_by_content(
                        user_input,
                        vectorizer,
                        tfidf_matrix,
                        companies_df,
                        top_n
                    )

                if result.empty:
                    st.info("ℹ️ Không tìm thấy công ty có độ tương đồng phù hợp với từ khóa của bạn.")
                else:
                    st.success(f"🎉 Đã tìm thấy **{len(result)} công ty** phù hợp nhất với nhu cầu của bạn:")
                    for _, row in result.iterrows():
                        display_company_card(row)

    elif search_mode == "🏢 Theo tên công ty":
        st.markdown("<div style='margin-bottom: 8px; font-size: 0.82rem; color: #64748B;'>💡 Hệ thống sẽ tìm các công ty có quy mô và đặc thù tương tự công ty bạn chọn.</div>", unsafe_allow_html=True)

        with st.container(border=True, key="recommend_search_panel"):
            col_comp, col_num_c, col_btn_c = st.columns([4, 0.5, 1], vertical_alignment="bottom")
            with col_comp:
                selected_company = st.selectbox(
                    "Chọn công ty",
                    options=company_names_list,
                    key="recommend_selected_company",
                )
            with col_num_c:
                top_n_comp = st.selectbox("Số lượng", options=[5, 10, 15], index=0, key="recommend_company_top_n")
            with col_btn_c:
                run_comp_btn = st.button("🔎 Tìm công ty", type="primary", width="stretch", key="btn_run_comp_rec")

            selected_id = name_to_id.get(selected_company)
            target_company = id_to_company.get(selected_id)

            if target_company is not None:
                st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                st.markdown(f"**📌 Hồ sơ đang chọn: {selected_company}**")
                render_company_overview_card(target_company)

        if run_comp_btn:
            with st.spinner(f"Đang phân tích các đặc trưng tương đồng với {selected_company}..."):
                result = recommend_similar_company(
                    selected_id,
                    companies_df,
                    similarity_matrix,
                    top_n_comp
                )

            if result.empty:
                st.info("ℹ️ Không tìm thấy công ty tương đồng trong tập dữ liệu.")
            else:
                st.success(f"🎉 Top **{len(result)} công ty tương tự nhất** với **{selected_company}**:")
                for _, row in result.iterrows():
                    display_company_card(row)

# =========================================================
# 9. TRANG 3: DỰ ĐOÁN ĐÁNH GIÁ & SENTIMENT AI
# =========================================================

elif selected_menu == "📊 Dự đoán đánh giá":
    st.markdown("""
    <div class="itv-hero-badge">📊 Company Review & Sentiment</div>
    <h1 style="margin-bottom: 6px;">Đánh Giá Công Ty & Dự Đoán Xu Hướng</h1>
    <p style="color: #475569; font-size: 0.95rem; margin-bottom: 20px;">
        Tra cứu hồ sơ công ty, đánh giá từ nhân viên và dự đoán đánh giá có xu hướng <strong>Recommend</strong> hay <strong>Not Recommend</strong>.
    </p>
    """, unsafe_allow_html=True)

    with st.container(border=True, key="classification_search_panel"):
        col_label, col_sel, _ = st.columns([0.9, 2.8, 4], vertical_alignment="center")
        with col_label:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #1E293B;'>Chọn công ty</div>", unsafe_allow_html=True)

        with col_sel:
            selected_company = st.selectbox(
                "Tên công ty",
                options=company_names_list,
                key="classification_company",
                label_visibility="collapsed"
            )

        selected_id = name_to_id.get(selected_company)
        company = id_to_company.get(selected_id)

        st.markdown("<hr style='margin: 12px 0 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        render_company_dashboard_header(selected_company, company, selected_id)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    render_review_classification(selected_company, company)


# =========================================================
# 10. TRANG 4: GIỚI THIỆU DỰ ÁN
# =========================================================

elif selected_menu == "ℹ️ Giới thiệu dự án":
    st.markdown("""
    <div class="itv-hero-badge">ℹ️ About the Project</div>
    <h1 style="margin-bottom: 6px;">Giới Thiệu Dự Án</h1>
    """, unsafe_allow_html=True)

    with st.container(border=True, key="about_project_goal"):
        st.subheader("🎯 Mục Tiêu Dự Án")
        st.write(
            "Xây dựng hệ thống hỗ trợ tìm kiếm, khám phá và đánh giá các công ty công nghệ dựa trên dữ liệu. "
            "Hệ thống giải quyết bài toán: **Content-Based Filtering** và **Classification – “Recommend or Not”**."
        )


    with st.container(border=True, key="about_company_recommendation"):
        st.subheader("1. Company Recommendation")
        st.caption("Content-Based Filtering")
        st.write(
            "Chuyển đổi quốc gia, quy mô, mô tả công ty, kỹ năng công nghệ thành không gian vector đặc trưng sử dụng **TF-IDF Vectorizer**. "
            "Độ tương đồng giữa người dùng và các công ty được đo lường bằng **Cosine Similarity**, cho phép tìm kiếm nhanh và chính xác."
        )

    with st.container(border=True, key="about_sentiment_classification"):
        st.subheader("2. Review Sentiment Classification")
        st.caption("Logistic Regression với Threshold Tuning")
        st.write(
            "Dự đoán xác suất nhân viên sẽ khuyên bạn bè gia nhập công ty (**Recommend** vs **Not Recommend**). "
            "Kết hợp 8 đặc trưng (các điểm đánh giá khía cạnh, loại hình công ty, chế độ OT) và đặc trưng văn bản đã chuẩn hóa qua NLP."
        )
        st.markdown(f"""
        - Mô hình chiến thắng: **Logistic Regression** (vượt qua SVM, KNN, Random Forest)
        - Ngưỡng kích hoạt Class 0: **{clf_threshold:.2f}** (tối ưu hóa phân phối mất cân bằng)
        """)

    with st.container(border=True, key="about_model_performance"):
        st.subheader("📈 Hiệu Năng Mô Hình Phân Loại (Model Performance)")
        b1, b2, b3, b4 = st.columns(4)
        # SỬA 4 (CHƯA ĐỔI SỐ, CẦN KIỂM TRA LẠI): ba số dưới đây mâu thuẫn nhau: với Precision 79.38% và Recall 82.13%
        # thì F1 phải là ≈ 80.7%, không phải 76.81% (ngược lại P=79.38%, F1=76.81% suy ra R ≈ 74.4%).
        # Mô hình trên máy mỗi người khác nhau nên không tự sửa số được; hãy lấy lại P/R/F1 của lớp No
        # từ notebook của mô hình đang dùng (đúng với ngưỡng trong file .pkl) rồi điền vào đây.
        with b1:
            st.metric("Precision", "79.38%")
        with b2:
            st.metric("Recall", "82.13%")
        with b3:
            st.metric("F1-Score", "76.81%")
        with b4:
            st.metric("Ngưỡng Class 0", f"{clf_threshold:.2f}")

        st.caption("Mô hình đã được thử nghiệm và so sánh trên 4 thuật toán: Logistic Regression, Linear SVM, KNN và Random Forest trong môi trường Scikit-Learn.")


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown(f"""
<div class="app-footer">
    <div style="font-weight: 700; color: #1E293B; margin-bottom: 4px;">
        IT COMPANY RECOMMENDER SYSTEM
    </div>
    <div>
        Cơ sở dữ liệu: <strong>{len(companies_df)} Công ty</strong> & <strong>{total_reviews_count:,} Đánh giá</strong>
    </div>
</div>
""", unsafe_allow_html=True)
