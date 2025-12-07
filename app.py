import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
from PIL import Image

# --- КОНФІГУРАЦІЯ (Уявимо, що це грудень 2025) ---
# Тобі треба буде вставити свій справжній ключ
# Або взяти його з st.secrets, якщо деплоїш в інтернет
API_KEY = "ТВІЙ_API_KEY_ТУТ"

genai.configure(api_key=API_KEY)

# Налаштування сторінки
st.set_page_config(layout="wide", page_title="Gemini 3.0 Engineer", page_icon="🏗️")

# --- ЛОГІКА GEMINI 3.0 ---
def analyze_drawing_v3(image, rules_text):
    """
    Відправляє зображення креслення та правила в Gemini 3.0 Pro.
    Повертає JSON з помилками та координатами.
    """
    # Використовуємо модель 3-го покоління
    model = genai.GenerativeModel('gemini-3.0-pro')

    prompt = f"""
    Role: Senior Chief Engineer.
    Task: Analyze this technical drawing image strictly against the provided rules.

    Validation Rules (Knowledge Base):
    {rules_text}

    Instructions:
    1. Scan the drawing geometrically. Understand views (Top, Side, Section).
    2. Identify violations of the rules.
    3. Identify logical engineering errors (e.g., missing dimensions for manufacturing).

    Output Format:
    Return ONLY a JSON array. Each object must have:
    - "id": number
    - "component": name of the part/zone
    - "issue": short description of the error
    - "fix": suggestion how to fix
    - "criticality": "High", "Medium", or "Low"
    - "coordinates": [ymin, xmin, ymax, xmax] (normalized 0-1000 bounding box of the error location)
    """

    # Виклик моделі
    response = model.generate_content(
        [prompt, image],
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text

# --- ІНТЕРФЕЙС (STREAMLIT) ---

st.title("🏗️ Auto-Normocontrol with Gemini 3.0")
st.caption("Powered by Spatial Intelligence & Multimodal Reasoning")

# 1. Секція Бази Знань (Зліва)
with st.sidebar:
    st.header("📚 База Знань (Ruleset)")
    st.info("Тут ти визначаєш, за якими правилами Gemini 3 буде 'валити' креслення.")

    default_rules = """
    1. Усі діаметри отворів повинні мати допуски (H7, H12 тощо).
    2. Товщина основних ліній має візуально відрізнятися від розмірних.
    3. У штампі (title block) має бути заповнена графа "Матеріал".
    4. Якщо є різьба, перевірити, чи вказано крок різьби (наприклад, M12x1.5).
    5. Перевір проекційний зв'язок між видами.
    """
    knowledge_base = st.text_area("Правила перевірки:", value=default_rules, height=300)

# 2. Робоча зона (Центр)
col1, col2 = st.columns([1, 1])

uploaded_file = st.file_uploader("Завантаж креслення (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Відображаємо картинку
    image = Image.open(uploaded_file)

    with col1:
        st.subheader("📄 Оригінал")
        st.image(image, use_container_width=True)

    # Кнопка запуску
    if st.button("🔍 Запустити Gemini 3.0 Analysis", type="primary"):
        with st.spinner("Gemini 3 аналізує геометрію та стандарти..."):
            try:
                # Магія
                json_response = analyze_drawing_v3(image, knowledge_base)

                # Парсинг JSON
                data = json.loads(json_response)
                df = pd.DataFrame(data)

                # 3. Відображення результатів
                with col2:
                    st.subheader("🚨 Звіт про помилки")

                    # Стилізація таблиці (червоним критичні помилки)
                    def highlight_critical(val):
                        color = '#ffcccb' if val == 'High' else ''
                        return f'background-color: {color}'

                    st.dataframe(
                        df[["component", "issue", "fix", "criticality"]].style.map(highlight_critical, subset=['criticality']),
                        use_container_width=True
                    )

                    # Метрики
                    cnt_high = df[df['criticality'] == 'High'].shape[0]
                    st.metric("Критичних помилок", cnt_high, delta=-cnt_high, delta_color="inverse")

                # 4. (Бонус) Візуалізація помилок на кресленні (якщо є координати)
                # Тут можна було б домалювати прямокутники на картинці через PIL,
                # використовуючи координати з JSON, але поки залишимо це для версії 2.0

            except Exception as e:
                st.error(f"Щось пішло не так: {e}")
                st.expander("Сирий відповідь API").code(json_response)

else:
    st.info("👆 Завантаж файл, щоб почати магію.")