import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import tempfile
import os
import base64
import time
import glob
import re
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from io import BytesIO
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional

# ==========================================
# 🔐 КОНФІГУРАЦІЯ
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Drawing Review",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Функція для завантаження CSS
def hide_branding():
    st.markdown("""
        <style>
            /* [class^="..."] означає "клас, що починається з..." */
            div[class^="_profilePreview"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)
    # hide_styles = """
    #     <style>
    #         ._profilePreview_gzau3_63 {
    #             display: none !important;
    #         }
    #     </style>
    # """
    # st.markdown(hide_styles, unsafe_allow_html=True)

hide_branding()

# Константи
RULES_DIR = "rules"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Ініціалізація API
@st.cache_resource
def init_genai():
    """Ініціалізує Gemini API з кешуванням."""
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("❌ Не знайдено API ключ у secrets.toml")
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"❌ Помилка конфігурації API: {e}")
        st.stop()

init_genai()

# ==========================================
# 🛠️ ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================

def display_pdf(file_path: str) -> None:
    """Відображає PDF у браузері через iframe з фіксом для Chrome."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Додаємо MIME type та параметри для Chrome
        pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=0&scrollbar=1" 
                width="100%" 
                height="800" 
                type="application/pdf"
                style="border: none;">
                <p>Ваш браузер не підтримує вбудований перегляд PDF. 
                   <a href="data:application/pdf;base64,{base64_pdf}" download="drawing.pdf">Завантажте файл</a>
                </p>
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Додаємо кнопку завантаження як запасний варіант
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Якщо PDF не відображається - завантажте файл",
                data=pdf_bytes,
                file_name="drawing.pdf",
                mime="application/pdf",
                help="Chrome може блокувати вбудований перегляд PDF"
            )
    except Exception as e:
        st.error(f"❌ Помилка відображення PDF: {e}")
        # Фоллбек - кнопка завантаження
        try:
            with open(file_path, "rb") as f:
                st.download_button(
                    label="📥 Завантажити PDF",
                    data=f.read(),
                    file_name="drawing.pdf",
                    mime="application/pdf"
                )
        except:
            st.error("Неможливо завантажити файл")

def clean_json_text(text: str) -> str:
    """Очищає відповідь AI від зайвих символів markdown."""
    text = text.strip()
    # Видаляємо markdown блоки
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

@lru_cache(maxsize=32)
def load_json_file(file_path: str) -> Optional[Dict]:
    """Завантажує JSON файл з кешуванням."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.sidebar.error(f"❌ Помилка читання {Path(file_path).name}: {e}")
        return None

def load_rules_from_json(selected_files: List[str]) -> str:
    """Зчитує вибрані JSON файли і формує єдиний текстовий рядок правил."""
    combined_rules = ""
    for file_path in selected_files:
        data = load_json_file(file_path)
        if data:
            filename = Path(file_path).name
            combined_rules += f"\n--- SOURCE: {filename} ---\n"
            combined_rules += json.dumps(data, indent=2, ensure_ascii=False)
            combined_rules += "\n"
    return combined_rules

def to_excel(df: pd.DataFrame) -> bytes:
    """Конвертує DataFrame в Excel файл у пам'яті."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis Results')
        # Автоширина колонок
        worksheet = writer.sheets['Analysis Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).str.len().max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
    return output.getvalue()

def upload_file_with_retry(file_path: str, mime_type: str = "application/pdf"):
    """Завантажує файл в Gemini з повторними спробами."""
    for attempt in range(MAX_RETRIES):
        try:
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            
            # Чекаємо обробки
            timeout = 60  # секунд
            start_time = time.time()
            while uploaded_file.state.name == "PROCESSING":
                if time.time() - start_time > timeout:
                    raise TimeoutError("Перевищено час очікування обробки файлу")
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                raise ValueError("Не вдалося обробити PDF на стороні Google")
            
            return uploaded_file
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"⚠️ Спроба {attempt + 1}/{MAX_RETRIES} не вдалася. Повтор через {RETRY_DELAY}с...")
                time.sleep(RETRY_DELAY)
            else:
                raise e

def analyze_pdf_drawing(file_path: str, rules_text: str, model_name: str) -> str:
    """Відправляє PDF в Gemini і повертає текст відповіді."""
    
    # Завантаження файлу
    with st.spinner("📤 Завантаження файлу..."):
        uploaded_file_ref = upload_file_with_retry(file_path)
    
    # Підготовка промпту
    prompt = f"""
Role: Lead Quality Control Engineer with expertise in technical documentation standards.

Task: Analyze this technical drawing PDF against the specific Ruleset provided below. 
Identify ALL violations, inconsistencies, and areas of non-compliance.

ACTIVE RULESET (Strictly follow these requirements):
{rules_text}

Instructions:
1. Check each page systematically
2. Identify specific components with issues
3. Provide actionable fix recommendations
4. Assess criticality (High/Medium/Low)

Output Format:
Return ONLY a valid JSON array. No markdown, no explanations.
Example:
[
    {{
        "page": 1,
        "component": "Shaft Detail A",
        "issue": "Missing tolerance specification per ISO 286-1",
        "fix": "Add tolerance class h7 to diameter 25mm",
        "criticality": "High"
    }}
]

If no issues found, return: []
"""
    
    # Виклик моделі
    model = genai.GenerativeModel(model_name)
    
    with st.spinner("🤖 AI аналізує креслення..."):
        response = model.generate_content(
            [prompt, uploaded_file_ref],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,  # Низька температура для точності
            }
        )
    
    # Видалення файлу з сервера Google
    try:
        genai.delete_file(uploaded_file_ref.name)
    except:
        pass
    
    return response.text

# ==========================================
# 🛠️ ФУНКЦІЇ ДЛЯ ПОРІВНЯННЯ CSV
# ==========================================

def parse_csv_value(value_str: str) -> str:
    """Парсить значення з JSON-like формату CSV."""
    try:
        # Видаляємо зайві лапки та парсимо JSON
        cleaned = value_str.strip('"')
        if cleaned.startswith('{"Value":'):
            data = json.loads(cleaned)
            return data.get("Value", "")
        return cleaned
    except:
        return value_str

def load_csv_file(uploaded_file) -> pd.DataFrame:
    """Завантажує CSV файл і парсить його структуру."""
    try:
        # Читаємо CSV
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
        
        # Парсимо кожну колонку
        for col in df.columns:
            df[col] = df[col].apply(parse_csv_value)
        
        return df
    except Exception as e:
        st.error(f"❌ Помилка читання CSV: {e}")
        return None

def extract_date_from_filename(filename: str) -> str:
    """Витягує дату з назви файлу."""
    # Шукаємо паттерн дати в форматі YYYY-MM-DD або подібному
    match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', filename)
    if match:
        return match.group(1)
    return ""

def compare_csv_files(df1: pd.DataFrame, df2: pd.DataFrame, file1_name: str, file2_name: str) -> Dict:
    """Порівнює два CSV файли і повертає різниці."""
    
    # Визначаємо, який файл новіший
    date1 = extract_date_from_filename(file1_name)
    date2 = extract_date_from_filename(file2_name)
    
    if date2 > date1:
        old_df, new_df = df1, df2
        old_name, new_name = file1_name, file2_name
    else:
        old_df, new_df = df2, df1
        old_name, new_name = file2_name, file1_name
    
    # Використовуємо full_path як ключ для порівняння
    old_paths = set(old_df['full_path'].values)
    new_paths = set(new_df['full_path'].values)
    
    # Знаходимо різниці
    added_files = new_paths - old_paths
    deleted_files = old_paths - new_paths
    common_files = old_paths & new_paths
    
    # Знаходимо змінені файли (де змінились дати)
    modified_files = []
    for path in common_files:
        old_row = old_df[old_df['full_path'] == path].iloc[0]
        new_row = new_df[new_df['full_path'] == path].iloc[0]
        
        if 'last_modif' in old_df.columns and 'last_modif' in new_df.columns:
            if old_row['last_modif'] != new_row['last_modif']:
                modified_files.append(path)
    
    return {
        'added': list(added_files),
        'deleted': list(deleted_files),
        'modified': modified_files,
        'old_name': old_name,
        'new_name': new_name
    }

def create_copy_button(file_path: str, key_suffix: str):
    """Створює кнопку для копіювання шляху файлу."""
    # Конвертуємо до Windows формату якщо потрібно
    windows_path = file_path.replace('/', '\\')
    
    # HTML для кнопки копіювання
    copy_btn = f"""
        <button onclick="navigator.clipboard.writeText('{windows_path}').then(() => {{
            this.innerHTML = '✅ Скопійовано!';
            setTimeout(() => {{ this.innerHTML = '📋 Копіювати'; }}, 2000);
        }})" 
        style="padding: 4px 12px; 
               background-color: #4CAF50; 
               color: white; 
               border: none; 
               border-radius: 4px; 
               cursor: pointer;
               font-size: 12px;
               margin-left: 10px;">
            📋 Копіювати
        </button>
    """
    
    st.markdown(f"{file_path} {copy_btn}", unsafe_allow_html=True)

# ==========================================
# 🖥️ ОСНОВНИЙ ІНТЕРФЕЙС
# ==========================================

# Ініціалізація session_state
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gemini-1.5-flash"  # Безпечний фоллбек

# --- Сайдбар ---
with st.sidebar:
    st.header("⚙️ Налаштування")
    
    # Завантаження доступних моделей
    @st.cache_data(ttl=3600)  # Кешуємо на 1 годину
    def get_available_models():
        """Отримує список доступних моделей з API."""
        try:
            models = genai.list_models()
            available = {}
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    # Створюємо читабельні назви
                    if 'gemini-3' in model_name.lower():
                        display_name = f"🔥 {model_name} (Gemini 3)"
                    elif 'gemini-2.5' in model_name.lower():
                        display_name = f"⚡ {model_name} (Gemini 2.5)"
                    elif 'gemini-2.0' in model_name.lower():
                        display_name = f"💨 {model_name} (Gemini 2.0)"
                    elif 'gemini-1.5-pro' in model_name.lower():
                        display_name = f"🎯 {model_name} (Gemini 1.5 Pro)"
                    elif 'gemini-1.5-flash' in model_name.lower():
                        display_name = f"⚡ {model_name} (Gemini 1.5 Flash)"
                    else:
                        display_name = model_name
                    
                    available[model_name] = display_name
            return available
        except Exception as e:
            st.error(f"Помилка завантаження моделей: {e}")
            # Фоллбек на базову модель
            return {"gemini-1.5-flash": "⚡ gemini-1.5-flash (Gemini 1.5 Flash)"}
    
    # Отримуємо доступні моделі
    available_models = get_available_models()
    
    # Вибір моделі
    if available_models:
        model_options = list(available_models.keys())
        
        # Визначаємо індекс за замовчуванням
        default_index = 0
        if st.session_state.selected_model in model_options:
            default_index = model_options.index(st.session_state.selected_model)
        elif 'gemini-2.5-flash' in model_options:
            default_index = model_options.index('gemini-2.5-flash')
        elif 'gemini-1.5-flash' in model_options:
            default_index = model_options.index('gemini-1.5-flash')
        
        selected_model = st.selectbox(
            "🤖 Модель AI:",
            options=model_options,
            format_func=lambda x: available_models[x],
            index=default_index,
            key="model_selector",
            help="Оберіть модель для аналізу креслень"
        )
        st.session_state.selected_model = selected_model
        
        # Показуємо кількість доступних моделей
        st.caption(f"📊 Доступно моделей: {len(available_models)}")
    else:
        st.error("❌ Не вдалося завантажити моделі")
        st.session_state.selected_model = "gemini-1.5-flash"
    
    st.divider()
    st.header("📚 Бібліотека Стандартів")
    
    # Завантаження файлів через інтерфейс
    st.caption("Завантажити додаткові файли:")
    uploaded_json_files = st.file_uploader(
        "Оберіть JSON файли",
        type=["json"],
        accept_multiple_files=True,
        help="Додаткові правила, які доповнять файли з папки rules/",
        key="json_uploader"
    )
    
    # Зберігаємо завантажені файли в session_state
    if uploaded_json_files:
        if 'uploaded_rules_files' not in st.session_state:
            st.session_state.uploaded_rules_files = {}
        
        for uploaded_file in uploaded_json_files:
            st.session_state.uploaded_rules_files[uploaded_file.name] = uploaded_file.getvalue()
    
    st.divider()
    
    # Збираємо всі доступні файли
    all_files = {}
    
    # 1. Локальні файли з папки rules/
    Path(RULES_DIR).mkdir(exist_ok=True)
    local_json_files = list(Path(RULES_DIR).glob("*.json"))
    for file_path in local_json_files:
        all_files[f"local:{file_path.name}"] = {
            "name": file_path.name,
            "path": str(file_path),
            "source": "📁 Локальні",
            "default": True  # Локальні файли включені за замовчуванням
        }
    
    # 2. Завантажені файли
    if 'uploaded_rules_files' in st.session_state:
        for file_name, file_content in st.session_state.uploaded_rules_files.items():
            # Створюємо тимчасовий файл
            temp_path = Path(tempfile.gettempdir()) / f"uploaded_{file_name}"
            temp_path.write_bytes(file_content)
            
            all_files[f"upload:{file_name}"] = {
                "name": file_name,
                "path": str(temp_path),
                "source": "☁️ Завантажені",
                "default": False  # Завантажені не включені за замовчуванням
            }
    
    # Відображення всіх файлів
    selected_files = []
    
    if all_files:
        st.caption("Оберіть файли для використання:")
        
        # Групуємо по джерелу для кращого відображення
        local_files = {k: v for k, v in all_files.items() if v["source"] == "📁 Локальні"}
        uploaded_files = {k: v for k, v in all_files.items() if v["source"] == "☁️ Завантажені"}
        
        # Локальні файли
        if local_files:
            st.markdown("**📁 Файли з папки rules/ (включені за замовчуванням):**")
            for key, file_info in local_files.items():
                if st.checkbox(
                    f"{file_info['name']}",
                    value=file_info['default'],
                    key=f"cb_{key}",
                    help=f"Джерело: {file_info['source']}"
                ):
                    selected_files.append(file_info['path'])
        
        # Завантажені файли
        if uploaded_files:
            st.markdown("**☁️ Завантажені файли:**")
            for key, file_info in uploaded_files.items():
                if st.checkbox(
                    f"{file_info['name']}",
                    value=file_info['default'],
                    key=f"cb_{key}",
                    help=f"Джерело: {file_info['source']}"
                ):
                    selected_files.append(file_info['path'])
    else:
        st.info("📂 Файлів не знайдено. Додайте JSON в папку rules/ або завантажте через форму вище")
    
    # Кнопка очистки завантажених файлів
    if 'uploaded_rules_files' in st.session_state and st.session_state.uploaded_rules_files:
        if st.button("🗑️ Очистити завантажені файли"):
            st.session_state.uploaded_rules_files = {}
            st.rerun()

    st.divider()
    
    # Попередній перегляд правил
    if selected_files:
        st.subheader("👀 Активні правила")
        with st.expander(f"🔍 {len(selected_files)} файл(ів) вибрано"):
            active_rules_preview = load_rules_from_json(selected_files)
            st.code(active_rules_preview, language="json")
        
        # Статистика
        total_chars = len(active_rules_preview)
        st.caption(f"📊 Розмір промпту: ~{total_chars:,} символів")
    else:
        st.warning("⚠️ Не вибрано жодного стандарту!")

# ==========================================
# ГОЛОВНЕ МЕНЮ (ТАБИ)
# ==========================================

main_tab1, main_tab2 = st.tabs(["🔍 Check", "⚖️ Compare"])

# ==========================================
# TAB 1: CHECK (існуючий функціонал)
# ==========================================
with main_tab1:
    uploaded_file = st.file_uploader(
        "📎 Завантаж PDF креслення",
        type=["pdf"],
        help="Максимальний розмір файлу залежить від налаштувань Streamlit",
        key="pdf_uploader"
    )

    # Очищаємо старі результати при завантаженні нового файлу
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.analysis_df = None
        st.session_state.last_uploaded_filename = uploaded_file.name

    if uploaded_file:
        # Зберігаємо тимчасовий файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Вкладки для інтерфейсу
        tab1, tab2 = st.tabs(["📄 Перегляд", "🤖 Аналіз"])
        
        with tab1:
            display_pdf(tmp_file_path)
        
        with tab2:
            st.subheader("🔍 Результат перевірки")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                btn_disabled = len(selected_files) == 0
                analyze_btn = st.button(
                    "🚀 Запустити перевірку",
                    type="primary",
                    disabled=btn_disabled,
                    use_container_width=True
                )
            with col2:
                if st.session_state.analysis_df is not None:
                    if st.button("🗑️ Очистити", use_container_width=True):
                        st.session_state.analysis_df = None
                        st.rerun()
            
            if analyze_btn:
                st.session_state.analysis_df = None
                
                try:
                    final_rules_text = load_rules_from_json(selected_files)
                    raw_response = analyze_pdf_drawing(
                        tmp_file_path,
                        final_rules_text,
                        st.session_state.selected_model
                    )
                    
                    json_response = clean_json_text(raw_response)
                    data = json.loads(json_response)

                    if not data:
                        st.success("✅ Чудово! AI не знайшов жодних порушень вибраних стандартів.")
                    else:
                        st.session_state.analysis_df = pd.DataFrame(data)
                        st.success(f"✅ Аналіз завершено. Знайдено {len(data)} невідповідностей.")

                except json.JSONDecodeError as e:
                    st.error(f"❌ Помилка парсингу JSON: {e}")
                    with st.expander("🐛 Debug Info"):
                        st.code(json_response if 'json_response' in locals() else raw_response)
                except Exception as e:
                    st.error(f"❌ Помилка аналізу: {e}")
                    with st.expander("🐛 Деталі помилки"):
                        st.exception(e)

            if btn_disabled:
                st.error("👈 Будь ласка, вибери хоча б один файл стандартів у меню зліва!")

            # Відображення результатів
            if st.session_state.analysis_df is not None:
                df = st.session_state.analysis_df
                
                # Метрики
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всього проблем", len(df))
                with col2:
                    high_count = len(df[df['criticality'] == 'High']) if 'criticality' in df.columns else 0
                    st.metric("Критичних", high_count, delta=None, delta_color="inverse")
                with col3:
                    unique_pages = df['page'].nunique() if 'page' in df.columns else 0
                    st.metric("Сторінок з проблемами", unique_pages)
                
                st.divider()
                
                # Налаштування таблиці
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_default_column(
                    resizable=True,
                    wrapText=True,
                    autoHeight=True,
                    sortable=True,
                    filter=True
                )
                
                # Налаштування колонок
                if 'issue' in df.columns:
                    gb.configure_column("issue", width=400)
                if 'fix' in df.columns:
                    gb.configure_column("fix", width=400)
                if 'component' in df.columns:
                    gb.configure_column("component", width=200)
                if 'page' in df.columns:
                    gb.configure_column("page", width=80)
                if 'criticality' in df.columns:
                    gb.configure_column("criticality", width=120)
                    
                    # Колір для criticality
                    jscode = JsCode("""
                    function(params) {
                        if (params.value === 'High') {
                            return {'color': 'white', 'backgroundColor': '#dc3545', 'fontWeight': 'bold'};
                        }
                        if (params.value === 'Medium') {
                            return {'color': 'black', 'backgroundColor': '#ffc107'};
                        }
                        if (params.value === 'Low') {
                            return {'color': 'black', 'backgroundColor': '#28a745', 'color': 'white'};
                        }
                        return {'color': 'black', 'backgroundColor': 'white'};
                    };
                    """)
                    gb.configure_column("criticality", cellStyle=jscode)
                
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                
                gridOptions = gb.build()

                grid_response = AgGrid(
                    df,
                    gridOptions=gridOptions,
                    height=500,
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False,
                    theme="streamlit",
                    key='analysis_grid',
                    reload_data=False
                )

                # Експорт
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    excel_data = to_excel(df)
                    st.download_button(
                        label="📥 Завантажити Excel",
                        data=excel_data,
                        file_name=f"analysis_{Path(uploaded_file.name).stem}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Завантажити CSV",
                        data=csv_data,
                        file_name=f"analysis_{Path(uploaded_file.name).stem}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        # Видаляємо тимчасовий файл при виході
        try:
            os.unlink(tmp_file_path)
        except:
            pass

    else:
        st.info("👆 Завантаж PDF креслення, щоб почати роботу")
        # Очищаємо стан при відсутності файлу
        st.session_state.analysis_df = None
        st.session_state.last_uploaded_filename = None

# ==========================================
# TAB 2: COMPARE (новий функціонал)
# ==========================================
with main_tab2:
    st.subheader("⚖️ Порівняння файлів CSV")
    st.caption("Завантажте два CSV файли для порівняння змін у файловій структурі")
    
    # Одна кнопка для завантаження 2х файлів
    csv_files = st.file_uploader(
        "📂 Оберіть 2 CSV файли для порівняння",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_files",
        # help="Виберіть рівно 2 CSV файли з однієї папки"
    )
    
    # Перевірка кількості файлів
    if csv_files and len(csv_files) == 2:
        csv_file1, csv_file2 = csv_files[0], csv_files[1]
        
        with st.spinner("🔄 Завантаження та порівняння файлів..."):
                # Завантажуємо обидва файли
                df1 = load_csv_file(csv_file1)
                df2 = load_csv_file(csv_file2)
                
                if df1 is not None and df2 is not None:
                    # Порівнюємо файли
                    comparison = compare_csv_files(df1, df2, csv_file1.name, csv_file2.name)
                    
                    # Відображаємо інформацію про файли
                    st.success("✅ Файли успішно порівняно!")
                    
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.info(f"📅 **Старіший файл:** {comparison['old_name']}")
                    with info_col2:
                        st.info(f"📅 **Новіший файл:** {comparison['new_name']}")
                
                st.divider()
                
                # Метрики
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("🆕 Нові файли", len(comparison['added']))
                with metric_col2:
                    st.metric("✏️ Змінені файли", len(comparison['modified']))
                with metric_col3:
                    st.metric("🗑️ Видалені файли", len(comparison['deleted']))
                
                st.divider()
                
                # Створюємо таби для результатів
                result_tab1, result_tab2, result_tab3 = st.tabs([
                    f"🆕 Нові файли ({len(comparison['added'])})",
                    f"✏️ Змінені файли ({len(comparison['modified'])})",
                    f"🗑️ Видалені файли ({len(comparison['deleted'])})"
                ])
                
                # Таб 1: Нові файли
                with result_tab1:
                    if comparison['added']:
                        st.subheader("Список нових файлів")
                        for idx, file_path in enumerate(sorted(comparison['added']), 1):
                            st.text(f"{idx}.")
                            st.code(file_path.replace('/', '\\'), language=None)
                    else:
                        st.info("✅ Нових файлів не знайдено")
                
                # Таб 2: Змінені файли
                with result_tab2:
                    if comparison['modified']:
                        st.subheader("Список змінених файлів")
                        for idx, file_path in enumerate(sorted(comparison['modified']), 1):
                            st.text(f"{idx}.")
                            st.code(file_path.replace('/', '\\'), language=None)
                    else:
                        st.info("✅ Змінених файлів не знайдено")
                
                # Таб 3: Видалені файли
                with result_tab3:
                    if comparison['deleted']:
                        st.subheader("Список видалених файлів")
                        for idx, file_path in enumerate(sorted(comparison['deleted']), 1):
                            st.text(f"{idx}.")
                            st.code(file_path.replace('/', '\\'), language=None)
                    else:
                        st.info("✅ Видалених файлів не знайдено")
    elif csv_files and len(csv_files) != 2:
        st.warning(f"⚠️ Потрібно вибрати рівно 2 файли. Зараз вибрано: {len(csv_files)}")
    else:
        st.info("👆 Оберіть 2 CSV файли для порівняння (можна вибрати обидва одночасно)")

# Футер
st.divider()
