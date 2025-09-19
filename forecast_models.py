# module/models/forecast_models.py

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict

# 다른 모듈에 있는 API 클라이언트 함수를 가져옵니다.
from module.api_clients.bigdata_service import fetch_metco_regn_visitr_dd_list

def predict_visitor_stats(
    area_cd: str, 
    target_date_str: str
) -> Optional[Dict[str, float]]:
    """
    특정 지역, 특정 날짜의 방문자 수를 예측합니다.
    (예측일 기준 1년 전의 한 달 치 데이터로, 같은 요일의 평균 방문자 수 계산)
    """
    try:
        target_date = datetime.strptime(target_date_str, '%Y%m%d')
    except ValueError:
        print(f"[Error] 날짜 형식이 잘못되었습니다. 'YYYYMMDD' 형식으로 입력해주세요.")
        return None

    # ✅ 1. 데이터 수집 기간 재설정: 목표일의 1년 전을 기준으로 약 한 달(30일)간
    end_date_lookback = target_date.replace(year=target_date.year - 1)
    start_date_lookback = end_date_lookback - timedelta(days=30)
    
    print(f"[{target_date_str} 예측] 데이터 수집 기간: {start_date_lookback.strftime('%Y%m%d')} ~ {end_date_lookback.strftime('%Y%m%d')}")

    past_data = fetch_metco_regn_visitr_dd_list(
        start_ymd=start_date_lookback.strftime('%Y%m%d'),
        end_ymd=end_date_lookback.strftime('%Y%m%d'),
        area_cd=area_cd 
    )

    if not past_data:
        print("[Error] 분석할 과거 데이터가 없습니다.")
        return None

    # 2. 데이터 처리: 수집한 데이터를 pandas DataFrame으로 변환
    df = pd.DataFrame(past_data)
    df['touNum'] = pd.to_numeric(df['touNum'])
    df['baseYmd'] = pd.to_datetime(df['baseYmd'], format='%Y%m%d')
    df['dayofweek'] = df['baseYmd'].dt.dayofweek
    
    # 3. 예측 로직: 목표 날짜와 같은 요일의 데이터만 필터링
    target_dayofweek = target_date.weekday()
    filtered_df = df[df['dayofweek'] == target_dayofweek]
    
    if filtered_df.empty:
        print(f"[Warning] 예측에 사용할 과거 요일 데이터가 부족합니다. (수집 기간: {start_date_lookback.strftime('%Y%m%d')} ~ {end_date_lookback.strftime('%Y%m%d')})")
        return None

    # 4. 결과 계산: 현지인과 외지인으로 나누어 평균 방문자 수 계산
    prediction = {}
    local_visitors = filtered_df[filtered_df['touDivCd'] == '1']['touNum'].mean()
    prediction['predicted_local_visitors'] = round(local_visitors, 2)
    non_local_visitors = filtered_df[filtered_df['touDivCd'] == '2']['touNum'].mean()
    prediction['predicted_non_local_visitors'] = round(non_local_visitors, 2)

    return prediction


if __name__ == '__main__':
    # 이 파일을 직접 실행했을 때 테스트할 코드
    print("🚀 방문자 수 예측 모델 테스트 시작...")

    # 예시: '지난주 월요일' 서울의 방문객 수 예측하기
    today = datetime.now()
    days_since_monday = (today.weekday() - 0 + 7) % 7
    last_monday = today - timedelta(days=days_since_monday + 7)
    last_monday_str = last_monday.strftime('%Y%m%d')
    
    # 서울(area_cd='11')의 지난주 월요일 방문객 예측
    result = predict_visitor_stats(
        area_cd='11', 
        target_date_str=last_monday_str
    )

    if result:
        print(f"\n[✅ 예측 결과] {last_monday_str} 서울특별시 예상 방문객")
        print(f" - 현지인: {result.get('predicted_local_visitors'):,.2f} 명")
        print(f" - 외지인: {result.get('predicted_non_local_visitors'):,.2f} 명")