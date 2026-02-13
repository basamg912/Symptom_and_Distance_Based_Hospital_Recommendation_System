from google import genai
import re
from project.utils import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
def extract_subject(text: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=text
        )
        extract_dept = client.models.generate_content(
            model="gemini-2.0-flash", contents=f"예상 진료과만 추출해줘.(특수문자, bold 등 텍스트 강조표현이 없는 실제 병원 진료과에 대한 plain text){response.text}"
        )
        extract_dept = extract_dept.text
        text = re.sub(r'[^\w\s가-힣]','',extract_dept)
        text = re.sub(r'\s+',' ',text)
        print(text)
        return text
    except Exception as e:
        print(f"Gemini 오류 : {e}")
        return text