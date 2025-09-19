# module/api_clients/durunubi.py

import pycurl
from io import BytesIO
import json
from urllib.parse import urlencode # pycurl은 URL 인코딩이 필요
from typing import List, Dict, Any, Optional # Optional 추가
import requests # unquote 기능을 위해 import

# config.py를 찾기 위해 프로젝트 루트 경로를 sys.path에 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import BASE_URLS, DURUNUBI_API_KEY

def _execute_pycurl_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """pycurl을 사용하여 API 요청을 실행하고 JSON 응답을 반환하는 내부 함수"""
    
    full_url = f"{endpoint}?{urlencode(params)}"
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, full_url)
    c.setopt(c.WRITEDATA, buffer)
    c.setopt(c.TIMEOUT, 30) # 두루누비는 데이터 양이 많을 수 있어 타임아웃을 30초로 설정
    
    try:
        print(f"[DEBUG] pycurl requesting: {full_url}") # 요청 URL 디버깅
        c.perform()
        status_code = c.getinfo(c.RESPONSE_CODE)
        if status_code != 200:
            print(f"[Error] durunubi_service: Request failed with status code {status_code} for URL: {full_url}")
            return None

        body = buffer.getvalue()
        # 응답 본문 디버깅 (필요시 사용, 매우 길 수 있음)
        # print(f"[DEBUG] pycurl response body: {body.decode('utf-8')[:500]}...") 
        return json.loads(body.decode('utf-8'))

    except pycurl.error as e:
        print(f"[Error] durunubi_service: pycurl request failed: {e} for URL: {full_url}")
        return None
    except json.JSONDecodeError as e:
        print(f"[Error] durunubi_service: JSONDecodeError: {e} for URL: {full_url}. Response body: {buffer.getvalue().decode('utf-8')[:500]}...")
        return None
    finally:
        c.close()

def _parse_api_response_items(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ API 응답에서 'item' 리스트를 안전하게 추출하는 헬퍼 함수 """
    if data is None:
        return []
    
    header = data.get("response", {}).get("header", {})
    # 두루누비 API는 성공 시 resultCode가 '00' 또는 '0000' 일 수 있음, 또는 아예 없을 수도 있음.
    # 실제 성공 응답의 header를 확인하여 정확한 조건 설정 필요.
    # 여기서는 우선 resultCode가 '00' 또는 '0000'이 아니면 오류로 간주하지 않고, items 유무로 판단.
    # print(f"[DEBUG] durunubi_service header: {header}") # 헤더 확인용

    items_node = data.get("response", {}).get("body", {}).get("items")
    if items_node is None: # 'items' 키 자체가 없는 경우 (데이터 없음 또는 다른 오류)
        if header.get("resultCode") not in ["00", "0000", None]: # resultCode가 명시적 오류를 나타내면 메시지 출력
             print(f"[Warning] durunubi_service API Error/No Data: {header.get('resultMsg')} (Code: {header.get('resultCode')})")
        return []

    # 'items'가 비어있는 문자열인 경우도 있음
    if isinstance(items_node, str) and not items_node.strip():
        return []
        
    # 'item'이 단일 객체 또는 리스트일 수 있음
    items = items_node.get("item", [])
    if isinstance(items, dict):
        return [items]
    return items if isinstance(items, list) else []


def fetch_routes(
    brd_div: Optional[str] = None,
    num_of_rows: int = 100, # API 기본 최대값이 100일 수 있음, 명세 확인 필요
    page_no: Optional[int] = None # 특정 페이지만 가져올 경우 사용
) -> List[Dict[str, Any]]:
    divisions = [brd_div] if brd_div else ["DNWW", "DNBW"] # 남파랑길, 두루누비 전체
    all_items: List[Dict[str, Any]] = []
    # 서비스 키에 포함된 특수문자가 이중 인코딩되는 것을 방지
    decoded_key = requests.utils.unquote(DURUNUBI_API_KEY)

    for div in divisions:
        current_page = page_no or 1
        while True:
            endpoint = f"{BASE_URLS['durunubi']}/routeList"
            params = {
                "serviceKey": decoded_key,
                "numOfRows":  num_of_rows,
                "pageNo":     current_page,
                "MobileOS":   "ETC",
                "MobileApp":  "TourRAG",
                "_type":      "json", # 응답 형식을 JSON으로 요청
                "brdDiv":     div
            }
            
            data = _execute_pycurl_request(endpoint, params)
            batch = _parse_api_response_items(data)
            
            if not batch: # 더 이상 가져올 데이터가 없으면 해당 division 종료
                break
            
            for item in batch: # 각 아이템에 brdDiv 정보 추가
                item["_brdDiv"] = div
            all_items.extend(batch)

            if page_no is not None: # page_no가 지정되었다면 한 페이지만 가져오고 종료
                break
            
            # 마지막 페이지 확인 로직 (API가 totalCount를 제공하면 그것을 사용하는 것이 더 정확)
            # 여기서는 가져온 아이템 수가 요청한 아이템 수보다 적으면 마지막 페이지로 간주
            if len(batch) < num_of_rows:
                break
            
            current_page += 1
            # 안전장치: 너무 많은 페이지를 요청하지 않도록 최대 페이지 제한 (예: 100페이지)
            if current_page > 100: 
                print(f"[Warning] Exceeded max page limit (100) for brdDiv: {div}")
                break
                
    return all_items

def fetch_courses(
    route_idx: str,
    brd_div: Optional[str] = None, # 특정 분기 코스만 가져올 경우
    num_of_rows: int = 100,
    page_no: Optional[int] = None
) -> List[Dict[str, Any]]:
    divisions = [brd_div] if brd_div else ["DNWW", "DNBW"] # 코스도 분기별로 나뉘어 있을 수 있음
    all_items: List[Dict[str, Any]] = []
    decoded_key = requests.utils.unquote(DURUNUBI_API_KEY)

    for div in divisions: # 실제 API가 brdDiv를 courseList에서 사용하는지 명세 확인 필요
        current_page = page_no or 1
        while True:
            endpoint = f"{BASE_URLS['durunubi']}/courseList"
            params = {
                "serviceKey": decoded_key,
                "routeIdx":   route_idx,
                "numOfRows":  num_of_rows,
                "pageNo":     current_page,
                "MobileOS":   "ETC",
                "MobileApp":  "TourRAG",
                "_type":      "json",
                "brdDiv":     div # 이 파라미터가 courseList에 유효한지 API 명세 확인!
                                  # 보통 routeIdx로 이미 상위 루트가 특정되므로 불필요할 수 있음.
            }

            data = _execute_pycurl_request(endpoint, params)
            batch = _parse_api_response_items(data)

            if not batch:
                break
            
            for item in batch:
                item["_brdDiv"] = div # 상위 분기 정보 추가
                start = item.get("crsStAddr", "").strip()
                end   = item.get("crsEndAddr", "").strip()
                item["region_address"] = f"{start} {end}".strip() # 메타데이터용 주소 필드
            all_items.extend(batch)

            if page_no is not None:
                break
            if len(batch) < num_of_rows:
                break
            current_page += 1
            if current_page > 100:
                print(f"[Warning] Exceeded max page limit (100) for routeIdx: {route_idx}, brdDiv: {div}")
                break
                
    return all_items