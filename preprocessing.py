import re
import numpy as np
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Load từ điển tiếng Anh sang tiếng Việt (ví dụ: "good" -> "tốt")
def load_english_dict(file_path):
    english_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                key, value = parts
                english_dict[key.lower()] = value.lower()
    return english_dict

english_dict = load_english_dict("data/english-vnmese.txt")

# 2. Từ điển teencode ngắn để giữ đúng ngữ nghĩa phủ định
TEENCODE_MAP = {
    "k": "không",
    "ko": "không",
    "khong": "không",
    "kg": "không",
    "cty": "công_ty",
    "dc": "được",
    "đc": "được",
    "ot": "làm_thêm_giờ",
}

# 3. Stopwords
# Danh sách các từ dừng cơ bản (không mang ngữ nghĩa phân loại)
BASE_VI_STOPWORDS = {
    "và", "là", "của", "có", "cho", "được", "trong", "với", "các", "những", "này", "đó", "khi", "để", "một",
    "cũng", "như", "thì", "mà", "ở", "tại", "từ", "đã", "sẽ", "đang", "nên", "bị", "vì", "nếu", "ra", "vào",
    "lên", "về", "theo", "do", "tôi", "mình", "em", "anh", "chị", "bạn", "còn", "hay", "nhưng", "cả", "đều",
    "cùng", "đến", "hoặc"
}
# Danh sách từ BẮT BUỘC GIỮ LẠI (Không được đưa vào stop words)
MUST_KEEP = {
    "không", "chưa", "chẳng", "chả", "không_hề",
    "rất", "quá", "lắm", "nhưng", "tuy", "tệ", "tốt"
}

# Tạo bộ VI_STOP hoàn chỉnh
VI_STOP = BASE_VI_STOPWORDS - MUST_KEEP

STOP_WORDS = VI_STOP | set(ENGLISH_STOP_WORDS)   # review có xen tiếng Anh nên bỏ cả stopword tiếng Anh

# 4. Xử lý từ đặc biệt
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

# 5. Cleaning function
def clean_review_a(text):
    """Chuẩn hóa một đoạn review; trả về chuỗi token cách nhau bởi khoảng trắng."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Chuyển chữ thường & Xóa URL / ký tự đặc biệt thừa
    text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())

    # 2. Quy đổi Teencode & Từ tiếng Anh (Xử lý cùng 1 lượt tách từ)
    # ĐÃ SỬA LỖI: Dùng text.split() để lặp qua từng TỪ thay vì từng KÝ TỰ
    words = []
    for w in text.split():
        # Đổi teencode trước
        word = TEENCODE_MAP.get(w, w)
        # Đổi tiếng Anh sang tiếng Việt
        word = english_dict.get(word, word)
        words.append(word)

    text = " ".join(words)

    # Chuẩn hóa khoảng trắng trước khi đưa vào PyVi
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tách từ ghép tiếng Việt bằng PyVi (Ví dụ: "công ty" -> "công_ty")
    text = ViTokenizer.tokenize(text)

    # 4. Xử lý từ đặc biệt (ví dụ: ghép cụm phủ định không_thích, không_tốt)
    text = process_special_word(text)

    #
    # text = process_postag_thesea(text)

    # 5. Lọc token chuẩn (giữ dấu _, bỏ số, bỏ câu/stop words)
    tokens = re.findall(r"\w+", text)
    cleaned_tokens = [
        t
        for t in tokens
        if len(t) > 1
        and not any(ch.isdigit() for ch in t)
        and t not in STOP_WORDS
    ]

    return " ".join(cleaned_tokens)

def prepare_data(df):
    df = df.copy()
    
    # Tiền xử lý cột text_clean
    df["title_clean"] = df["Title"].fillna("").astype(str).apply(clean_review_a)
    df["liked_clean"] = df["What I liked"].fillna("").astype(str).apply(clean_review_a)
    df["sugg_clean"] = df["Suggestions for improvement"].fillna("").astype(str).apply(clean_review_a)
    
    df["text_clean"] = (
        df["title_clean"] + " "
        + df["liked_clean"] + " "
        + df["sugg_clean"]
    ).str.strip()
    
    return df

def predict_with_class0_threshold(_df, _model, threshold_class0):
    '''Dự đoán review với ngưỡng xác suất của Class 0.'''
    
    # Dự đoán xác suất Recommend cho từng dòng review
    df = _df.copy()
    feature_cols = [
        "Rating", "Salary & benefits", "Training & learning",
        "Management cares about me", "Culture & fun", "Office & workspace",
        "Company Type", "Overtime Policy", "text_clean"
    ]
    X = df[feature_cols]
    
    # Tính xác suất Class 0 (Not Recommend)
    prob_class0 = _model.predict_proba(X)[:, 0]
    
    # Áp dụng quy tắc: Nếu prob_class0 >= th_class0 thì ra 0, ngược lại ra 1
    y_pred = np.where(prob_class0 >= threshold_class0, 0, 1)
    
    df["prob_class0"] = prob_class0
    df["prob_class1_recommend"] = 1 - prob_class0
    df["is_recommend"] = y_pred

    # Gom nhóm theo công ty
    summary = df.groupby(["Company Name", "Company Type", "Overtime Policy"]).agg(
        total_reviews=("text_clean", "count"),
        recommend_rate=("is_recommend", "mean"),
        avg_overall=("Rating", "mean"),
        avg_salary=("Salary & benefits", "mean"),
        avg_training=("Training & learning", "mean"),
        avg_mgmt=("Management cares about me", "mean"),
        avg_culture=("Culture & fun", "mean"),
        avg_workspace=("Office & workspace", "mean"),
    ).reset_index()

    summary["recommend_rate_pct"] = summary["recommend_rate"] * 100
    return df, summary
