# 파일: programming/module/api_clients/bigdata_service.py

import pycurl
from io import BytesIO
import json
from urllib.parse import urlencode
from typing import List, Dict, Any, Optional
import requests 

# config.py를 찾기 위해 프로젝트 루트 경로를 sys.path에 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import BASE_URLS, BIGDATA_API_KEY

def _execute_pycurl_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """pycurl을 사용하여 API 요청을 실행하고 JSON 응답을 반환하는 내부 함수"""
    
    # pycurl은 URL에 파라미터를 직접 인코딩해야 함
    full_url = f"{endpoint}?{urlencode(params)}"
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, full_url)
    c.setopt(c.WRITEDATA, buffer)
    c.setopt(c.TIMEOUT, 10) # 타임아웃 10초
    
    try:
        c.perform()
        # 요청 성공 여부 확인 (HTTP 상태 코드)
        status_code = c.getinfo(c.RESPONSE_CODE)
        if status_code != 200:
            print(f"[Error] Request failed with status code: {status_code}")
            return None

        body = buffer.getvalue()
        return json.loads(body.decode('utf-8'))

    except pycurl.error as e:
        # pycurl 예외 처리 (네트워크 레벨 오류 등)
        print(f"[Error] pycurl request failed: {e}")
        return None
    finally:
        c.close()

def fetch_metco_regn_visitr_dd_list(
    start_ymd: str, end_ymd: str, area_cd: Optional[str] = None, num_of_rows: int = 1000
) -> Optional[List[Dict[str, Any]]]:
    """광역 지자체 방문자 집계 조회 (pycurl 사용)"""
    endpoint = f"{BASE_URLS['bigdata_service']}/metcoRegnVisitrDDList"
    results: List[Dict[str, Any]] = []
    page_no = 1

    while True:
        print(f"[DEBUG] fetch_metco (pycurl) fetching page {page_no}...")
        # 서비스 키에 포함된 특수문자가 이중 인코딩되는 것을 방지
        decoded_key = requests.utils.unquote(BIGDATA_API_KEY)
        params = {
            "serviceKey": decoded_key, "startYmd": start_ymd, "endYmd": end_ymd,
            "numOfRows": num_of_rows, "pageNo": page_no, "MobileOS": "ETC",
            "MobileApp": "TourRAG", "_type": "json",
        }

        data = _execute_pycurl_request(endpoint, params)
        if data is None: return None # 요청 실패 시 None 반환

        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") != "0000":
            print(f"[Error] API Error on page {page_no}: {header.get('resultMsg')}")
            return None if page_no == 1 else results

        body = data.get("response", {}).get("body", {})
        if not body or not body.get("items"): break
        
        batch = body.get("items", {}).get("item", [])
        if not batch: break
        
        results.extend(batch)
        if body.get("numOfRows", 0) < num_of_rows or body.get("totalCount", 0) == len(results):
            break
        
        page_no += 1
    
    if area_cd:
        results = [r for r in results if str(r.get("areaCode")) == area_cd]

    return results

# fetch_locgo_regn_visitr_dd_list 함수도 pycurl을 사용하도록 수정해야 합니다.
# 우선 위의 함수가 성공하는지 확인하는 것이 중요합니다.