import pandas as pd
import re
import os
from datetime import date

# =========================
# Curățare radicală pentru limbaje C-like
# =========================
def clean_c_like_hard(code: str) -> str:
    if not isinstance(code, str):
        return ""
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if any(x in line for x in ['//', '/*', '*/', '#']) or re.match(r'^\s*\*', line):
            continue
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    return re.sub(r'\s+', ' ', code).strip()

def clean_python(code: str) -> str:
    if not isinstance(code, str):
        return ""
    code = re.sub(r'"""[\s\S]*?"""', '', code, flags=re.MULTILINE)
    code = re.sub(r"'''[\s\S]*?'''", '', code, flags=re.MULTILINE)
    lines = []
    for line in code.split('\n'):
        if re.match(r'^\s*#', line):
            continue
        if '#' in line:
            line = line.split('#')[0]
        lines.append(line)
    code = '\n'.join(lines)
    return re.sub(r'\s+', ' ', code).strip()

def clean_php(code: str) -> str:
    if not isinstance(code, str):
        return ""
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if any(x in line for x in ['//', '/*', '*/', '#']) or re.match(r'^\s*\*', line):
            continue
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    return re.sub(r'\s+', ' ', code).strip()

def clean_code(code: str, language: str) -> str:
    lang = language.lower() if isinstance(language, str) else ""
    if lang == "python":
        return clean_python(code)
    elif lang in ["c", "cpp", "c++", "csharp", "java", "go", "js", "javascript"]:
        return clean_c_like_hard(code)
    elif lang == "php":
        return clean_php(code)
    else:
        return code.strip()

# =========================
# Feature extraction
# =========================
def extract_features(code: str) -> dict:
    tokens = re.findall(r'[A-Za-z_]\w*', code)
    unique_tokens = set(tokens)
    return {
        'length': len(code),
        'num_lines': code.count('\n') + 1 if code else 0,
        'num_keywords': sum(code.count(kw) for kw in
                            ['def','class','import','return','if','for','while',
                             'public','static','void','function','package']),
        'num_special_chars': sum(code.count(c) for c in ['{','}','(',')','[',']']),
        'avg_indent': (sum(len(line) - len(line.lstrip()) for line in code.split('\n'))
                       / max(1, code.count('\n')+1)) if code else 0.0,
        'token_count': len(tokens),
        'unique_token_count': len(unique_tokens),
        'token_diversity': (len(unique_tokens) / max(1, len(tokens))) if tokens else 0.0
    }

# =========================
# Raport de diferențe
# =========================
def save_diff_report(df, original_path):
    df['modificat'] = df['code'] != df['code_clean']
    df_diff = df[df['modificat']]
    num_modificate = len(df_diff)
    total = len(df)
    print(f"🔍 {num_modificate} din {total} coduri au fost modificate ({(num_modificate / total * 100) if total else 0:.2f}%)")

    folder = os.path.dirname(original_path)
    base = os.path.basename(original_path).replace(".parquet", "")
    report_path = os.path.join(folder, f"{base}_diff_report.csv")
    df_diff[['code', 'code_clean']].to_csv(report_path, index=False)
    print(f"📄 Raport salvat: {report_path}")

# =========================
# Preprocesare fișier
# =========================
def preprocess_file(path: str):
    try:
        df = pd.read_parquet(path)
        df['code_clean'] = df.apply(lambda row: clean_code(row['code'], row.get('language', 'c')), axis=1)

        features = df['code_clean'].apply(extract_features)
        features_df = pd.DataFrame(features.tolist())
        df.reset_index(drop=True, inplace=True)
        for col in features_df.columns:
            df[col] = features_df[col]

        today = date.today().isoformat()
        out_path = path.replace(".parquet", f"_preprocessed_{today}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Preprocesat: {os.path.basename(path)} → {os.path.basename(out_path)}")

        save_diff_report(df, path)

    except Exception as e:
        print(f"⚠ Eroare la fișierul {path}: {e}")

# =========================
# Rulează pe task_A și task_B
# =========================
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    for folder in ["task_A", "task_B"]:
        print(f"\n🔄 Preprocesare în {folder}...")
        full_path = os.path.join(base_dir, folder)
        if os.path.exists(full_path):
            for file in os.listdir(full_path):
                if file.endswith(".parquet"):
                    preprocess_file(os.path.join(full_path, file))
        else:
            print(f"❌ Folderul {folder} nu există.")

    print("\n✅ Preprocesarea completă s-a încheiat.")

