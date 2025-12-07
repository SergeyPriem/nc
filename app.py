import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import tempfile
import os
import base64
import time
import glob
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
    page_title="AI Drawing Engineer Pro",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

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
    """Відображає PDF у браузері через iframe."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Помилка відображення PDF: {e}")

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
# 🖥️ ОСНОВНИЙ ІНТЕРФЕЙС
# ==========================================

st.title("🏗️ AI Нормоконтроль: Modular Edition")

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
    
    # Вибір джерела файлів
    source_option = st.radio(
        "Джерело файлів правил:",
        ["📁 Локальні файли (rules/)", "☁️ Завантажити файли"],
        help="Оберіть звідки брати JSON файли зі стандартами"
    )
    
    selected_files = []
    
    if source_option == "📁 Локальні файли (rules/)":
        st.caption("Файли з папки rules/:")
        
        # Створюємо папку rules якщо її немає
        Path(RULES_DIR).mkdir(exist_ok=True)

        json_files = list(Path(RULES_DIR).glob("*.json"))

        if not json_files:
            st.info("📂 Файлів не знайдено. Додай JSON в папку rules/ або використай завантаження")
        else:
            for file_path in json_files:
                file_name = file_path.name
                if st.checkbox(f"📄 {file_name}", value=False, key=f"cb_local_{file_name}"):
                    selected_files.append(str(file_path))
    
    else:  # Завантаження файлів
        st.caption("Завантажте JSON файли зі стандартами:")
        
        uploaded_json_files = st.file_uploader(
            "Оберіть JSON файли",
            type=["json"],
            accept_multiple_files=True,
            help="Можна вибрати декілька файлів одночасно"
        )
        
        if uploaded_json_files:
            # Зберігаємо завантажені файли тимчасово
            if 'uploaded_rules_files' not in st.session_state:
                st.session_state.uploaded_rules_files = {}
            
            for uploaded_file in uploaded_json_files:
                # Зберігаємо в session_state
                st.session_state.uploaded_rules_files[uploaded_file.name] = uploaded_file.getvalue()
            
            # Чекбокси для вибору
            st.caption("Оберіть файли для використання:")
            for file_name in st.session_state.uploaded_rules_files.keys():
                if st.checkbox(f"📄 {file_name}", value=True, key=f"cb_upload_{file_name}"):
                    # Створюємо тимчасовий файл
                    temp_path = Path(tempfile.gettempdir()) / file_name
                    temp_path.write_bytes(st.session_state.uploaded_rules_files[file_name])
                    selected_files.append(str(temp_path))
        else:
            st.info("👆 Завантажте JSON файли з вашого комп'ютера")

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

# --- Головна частина ---
uploaded_file = st.file_uploader(
    "📎 Завантаж PDF креслення",
    type=["pdf"],
    help="Максимальний розмір файлу залежить від налаштувань Streamlit"
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

# Футер
st.divider()
st.caption("🏗️ AI Drawing Engineer Pro | Powered by Google Gemini")
