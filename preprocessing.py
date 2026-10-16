import re
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import Optional

def clean_text(text: object, stop_words=None) -> str:
    """
    Chuẩn bị văn bản cho TF-IDF.

    - Xử lý missing values và lowercase.
    - Chuẩn hóa dấu nháy Unicode.
    - Chuẩn hóa sở hữu cách: team's -> team.
    - Giữ từ phủ định: don't, doesn't, can't...
    - Chuẩn hóa từ ghép: back-end -> back_end.
    - Giữ thuật ngữ: C, C++, C#, R&D, CI/CD, Node.js...
    - Giữ các cụm từ quan trọng như:
      C language, A to Z, Ho Chi Minh City.
    - Loại stop words nhưng giữ từ phủ định và từ được bảo vệ.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    # 1. Chuẩn hóa dấu nháy Unicode
    text = re.sub(r"[\u2018\u2019\u02bc\u0060]", "'", text)

    # Loại bỏ contraction phổ biến
    text = re.sub(
        r"\b(?:"
        r"i|you|we|they|he|she|it|that|there|who|what|"
        r"here|how|where|when|why|let"
        r")'(?:s|re|ve|ll|d|m|t)\b",
        " ",
        text
    )
    # 2. Bảo vệ một số cụm từ
    protected_phrases = {
        r"\ba\s+to\s+z\b": "a_to_z",
        r"\bho\s+chi\s+minh\s+city\b": "ho_chi_minh_city",
        r"\bviet\s+nam\b": "viet_nam"
    }

    for pattern, replacement in protected_phrases.items():
        text = re.sub(pattern, replacement, text)

    # 3. Chuẩn hóa ngôn ngữ C, không ảnh hưởng C++ và C#
    text = re.sub(r"\bc(?:\s+language|[_-]language)?\b(?![+#])", "c_language", text)

    # 4. Chuẩn hóa sở hữu cách: team's -> team; don't vẫn giữ nguyên.
    text = re.sub(r"\b([a-z0-9_]+)'s\b", r"\1", text)

    # 5. Chuẩn hóa từ ghép:back-end -> back_end; end-to-end -> end_to_end.
    text = re.sub(r"(?<=\w)-(?=\w)", "_", text)

    # 6. Chuẩn hóa khoảng trắng
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Nhận diện thuật ngữ đặc biệt
    special_terms = [
        r"u\.s\.a\.",
        r"u\.s\.",
        r"asp\.net",
        r"node\.js",
        r"ci/cd",
        r"r&d_ing",
        r"r&d",
        r"f&b",
        r"p&g",
        r"c\+\+",
        r"c#",
        r"\.net",
    ]

    special_terms.sort(key=len, reverse=True)               # Sắp xếp cụm dài trước để ưu tiên khớp cụm đầy đủ

    pattern = (
        r"(?<!\w)(?:"
        + "|".join(special_terms)
        + r")(?!\w)"
        + r"|[a-z0-9]+(?:'[a-z]+)+"  # Contractions
        + r"|[a-z0-9]+(?:_[a-z0-9]+)+"  # Cụm bảo vệ/từ ghép
        + r"|\b[a-z0-9]+\b"  # Token thông thường
    )

    words = re.findall(pattern, text)

    # 8. Stop words: mặc định dùng bộ tiếng Anh của sklearn
    stop_words = stop_words or set(ENGLISH_STOP_WORDS)
    stop_words = {word.lower() for word in stop_words}

    # 9. Các token được bảo vệ
    protected_tokens = {
        "c_language",
        "a_to_z",
        "ho_chi_minh_city",
        "c++",
        "c#",
        "r&d",
        "r&d_ing",
        "f&b",
        "p&g",
        "ci/cd",
        "node.js",
        "asp.net",
        ".net",
        "u.s.",
        "u.s.a.",
    }

    # 10. Lọc stop words và token rác
    tokens = [
        word for word in words
        if (
            word not in stop_words
            and not word.isdigit()
        ) or word in protected_tokens
    ]

    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    processed_column: str = "content_processed",
    stop_words: Optional[set] = None
) -> pd.DataFrame:
    """Tạo cột text đã được preprocessing."""
    result = df.copy()

    result[processed_column] = result[text_column].apply(
        lambda text: clean_text(text, stop_words)
    )

    return result



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
