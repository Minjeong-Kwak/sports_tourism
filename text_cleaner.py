# programming/module/preprocessing/text_cleaner.py

import re
import html
from typing import Dict, Any

def clean_html(raw_html: str) -> str:
    """
    HTML 태그 제거 및 HTML 엔티티 디코딩.
    """
    text = html.unescape(raw_html or "")
    text = re.sub(r'<[^>]+>', ' ', text)      # 태그 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 공백 정리
    return text

def clean_route_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    두루누비 routeList/item 구조 전처리
    """
    return {
        'route_idx':   item.get('routeIdx'),
        'theme_name':  item.get('themeNm'),
        'linemsg':     clean_html(item.get('linemsg')),
        'description': clean_html(item.get('themedescs')),
        'brd_div':     item.get('brdDiv'),
    }

def clean_course_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    두루누비 courseList/item 구조 전처리
    """
    return {
        'route_idx':   item.get('routeIdx'),
        'course_idx':  item.get('crsIdx'),
        'course_name': item.get('crsKorNm'),
        'level':       item.get('crsLevel'),
        'distance':    item.get('crsDstnc'),
        'contents':    clean_html(item.get('crsContents')),
        'summary':     clean_html(item.get('crsSummary')),
    }

def clean_forecast_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    방문자 추이 예측(item) 전처리
    """
    return {
        'area_code':     item.get('areaCd'),
        'sigun_code':    item.get('signguCd'),
        'spot_name':     item.get('tAtsNm'),
        'date':          item.get('baseYmd'),
        'predicted_num': float(item.get('predictedNum', 0) or 0),
    }
