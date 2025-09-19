# programming/module/api_clients/weather_gemini.py

import google.generativeai as genai
from typing import Optional, Dict, Any
import json
import re # JSON 유사 문자열 추출을 위해 추가
from datetime import datetime, timedelta

# config.py를 찾기 위해 프로젝트 루트 경로를 sys.path에 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import GEMINI_API_KEY

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY가 .env 파일에 설정되지 않았거나 config.py에서 로드되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)

# 사용할 Gemini 모델 설정
# 최신 모델이나 사용 가능한 모델로 변경 가능 (예: 'gemini-1.5-flash', 'gemini-1.0-pro')
GEMINI_MODEL_NAME = 'gemini-1.5-flash' # 또는 'gemini-pro'

def get_weather_forecast_from_gemini(
    location: str,          # 예: "서울", "부산 해운대구", "제주도 서귀포시"
    target_date_str: str    # 'YYYYMMDD' 형식
) -> Optional[Dict[str, Any]]:
    """
    Gemini API를 사용하여 특정 지역과 날짜의 날씨 예보를 JSON 형식으로 요청하고 받아옵니다.
    (예상 최저/최고 기온, 날씨 상태, 강수 확률 등)
    """
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    # Gemini에게 JSON 형식으로 답변을 요청하는 프롬프트
    prompt = f"""
    {target_date_str} 날짜의 {location} 지역 날씨 예보를 다음 JSON 형식으로 제공해줘:
    {{
        "date": "{target_date_str}",
        "location": "{location}",
        "overall_weather": "날씨 상태 요약 (예: 대체로 맑음, 구름 조금 후 비)",
        "min_temp_celsius": "예상 최저 기온 (섭씨, 숫자만)",
        "max_temp_celsius": "예상 최고 기온 (섭씨, 숫자만)",
        "precipitation_probability_percent": "강수 확률 (백분율, 숫자만, 정보 없으면 null)",
        "humidity_percent": "예상 습도 (백분율, 숫자만, 정보 없으면 null)",
        "wind_speed_mps": "예상 풍속 (m/s, 숫자만, 정보 없으면 null)",
        "brief_comment": "날씨에 대한 간략한 코멘트 (예: 활동하기 좋은 날씨입니다, 우산 필요)"
    }}
    위 형식에서 '값' 부분만 채워주고, 만약 특정 항목의 정보가 없다면 그 값은 null로 표시해줘.
    숫자 값은 반드시 숫자만 포함하고, 단위는 이미 명시되어 있으니 값에 포함하지 마.
    답변은 JSON 객체만 포함해야 하며, 다른 설명이나 대화는 제외해줘.
    """

    print(f"▶ DEBUG (Gemini) Weather forecast prompt for {location} on {target_date_str} sent.")

    try:
        response = model.generate_content(prompt)
        
        # Gemini 응답에서 JSON 부분만 추출 시도
        response_text = response.text.strip()
        
        # 응답이 마크다운 코드 블록(```json ... ```)으로 감싸져 있을 경우 추출
        match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
        if match:
            json_str = match.group(1).strip()
        else:
            # 코드 블록이 없다면, 텍스트 자체가 JSON이거나 JSON을 포함하는지 확인
            # 가장 간단하게는 첫 '{'와 마지막 '}'를 기준으로 잘라볼 수 있음 (완벽하지 않음)
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = response_text[first_brace : last_brace+1]
            else:
                json_str = response_text # 일단 그대로 사용

        print(f"▶ DEBUG (Gemini) Raw response text: {response_text}")
        print(f"▶ DEBUG (Gemini) Extracted JSON string: {json_str}")

        # 추출된 문자열을 JSON으로 파싱
        weather_data = json.loads(json_str)
        
        # 간단한 유효성 검사 (예: 'date'와 'location' 키가 있는지)
        if "date" not in weather_data or "location" not in weather_data:
            print(f"[Error] Gemini response JSON is missing required keys 'date' or 'location'.")
            return None
            
        return weather_data

    except Exception as e:
        print(f"[Error] Gemini API request or JSON parsing failed: {e}")
        print(f"   Gemini raw response was: {response.text if 'response' in locals() else 'No response object'}")
        return None

if __name__ == '__main__':
    # 이 파일을 직접 실행하여 테스트할 때 사용
    print("--- weather_gemini.py 직접 테스트 시작 ---")
    
    # 테스트할 지역 및 날짜
    test_location = "서울 중구"
    # 내일 날짜로 테스트
    tomorrow_dt = datetime.now() + timedelta(days=1)
    test_date = tomorrow_dt.strftime('%Y%m%d')
    
    print(f"\n--- {test_location}의 {test_date} 날씨 예보 조회 시도 ---")
    forecast = get_weather_forecast_from_gemini(
        location=test_location,
        target_date_str=test_date
    )
    
    if forecast:
        print("\n[✅ Gemini API 응답 성공]")
        print(json.dumps(forecast, indent=2, ensure_ascii=False))
    else:
        print("\n[❌ Gemini API 응답 실패 또는 파싱 오류]")