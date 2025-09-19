import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re

base_dir = Path(__file__).parent
load_dotenv(dotenv_path=base_dir / ".env")

from module.api_clients.durunubi import fetch_routes, fetch_courses
from module.preprocessing.text_cleaner import clean_course_item
from module.embedding.embedder import embed_texts
from module.vector_store.faiss_store import FaissStore
from openai import OpenAI
from module.rag.pipeline import RAGPipeline, CITY_TO_AREACD_FOR_PREDICTION
# from module.rag.pipeline import RAGPipeline # build_full_index에서는 직접 사용 안함

def build_full_index():
    print("\n[PROCESS] 전체 코스 인덱스 빌드 시작")
    
    idx_path   = base_dir / "faiss_index_all.pkl"
    idmap_path = base_dir / "id_map_all.pkl"
    meta_file_path = base_dir / "routes_meta.pkl"

    # --- 헬퍼 함수: 코스 데이터에서 지역 정보 및 주요 특징 키워드 추출 ---
    def extract_info_for_embedding(item_data: dict, course_name_cleaned: str) -> dict:
        # 이전에 정의했던 CITY_TO_AREACD_FOR_PREDICTION와 유사한 정보 활용 또는 직접 정의
        # (main.py가 pipeline.py를 import하지 않으므로, 필요시 이 딕셔너리를 config.py 등으로 옮겨 공용 사용 고려)
        KOREA_MAJOR_REGIONS = {
            "서울": ["서울특별시", "서울시"], "부산": ["부산광역시"], "대구": ["대구광역시"],
            "인천": ["인천광역시"], "광주": ["광주광역시"], "대전": ["대전광역시"],
            "울산": ["울산광역시"], "세종": ["세종특별자치시"],
            "경기": ["경기도"], "강원": ["강원특별자치도", "강원도"], "충북": ["충청북도"],
            "충남": ["충청남도"], "전북": ["전북특별자치도", "전라북도"], "전남": ["전라남도"],
            "경북": ["경상북도"], "경남": ["경상남도"], "제주": ["제주특별자치도"]
        }
        # 주요 시/군/구 및 기타 지역 관련 키워드 (필요에 따라 확장)
        SIGUNGU_AND_FEATURE_KEYWORDS = [
            "수원", "용인", "성남", "고양", "창원", "청주", "천안", "전주", "포항", "김해", "평택", "안산", "안양", "파주", "양평", "강화", "해남", "고성", "통영", "기장",
            "종로", "중구", "강남", "서초", "송파", "영등포", "마포", "해운대", "장안구", "권선구", "팔달구", "영통구",
            "북한산", "설악산", "지리산", "한라산", "남산", "광교산", "관악산", "호미곶", "땅끝마을", "소래습지", "인천대공원", "DMZ"
        ]
        
        extracted_info = {
            "sido": "",          # 예: 경기, 서울
            "sigungu": "",       # 예: 수원시, 종로구
            "detail_loc": "",    # 예: 장안구, 북한산
            "keywords": [],      # 추출된 모든 지역/특징 키워드
            "best_address": ""   # 대표 주소 또는 정보
        }

        # 정보 추출 대상: 코스명, 요약, 개요, 관광정보, 출발/도착지 주소, 시군명칭 등
        text_pool = (
            f"{course_name_cleaned} "
            f"{item_data.get('crsSummary', '')} "
            f"{item_data.get('crsContents', '')} "
            f"{item_data.get('crsTourInfo', '')} "
            f"{item_data.get('crsStAddr', '')} {item_data.get('crsEndAddr', '')} "
            f"{item_data.get('sigunNm', '')}"
        ).lower() # 검색을 위해 소문자로

        # 1. 시/도 추출
        for rep_name, aliases in KOREA_MAJOR_REGIONS.items():
            for alias in aliases:
                if alias.lower() in text_pool:
                    extracted_info["sido"] = rep_name
                    if rep_name not in extracted_info["keywords"]:
                        extracted_info["keywords"].append(rep_name)
                    break
            if extracted_info["sido"]:
                break
        
        # 2. 시/군/구 및 주요 지명 키워드 추출
        # SIGUNGU_AND_FEATURE_KEYWORDS 목록을 사용하여 text_pool에서 키워드 검색
        temp_keywords = []
        for kw in SIGUNGU_AND_FEATURE_KEYWORDS:
            if kw.lower() in text_pool:
                temp_keywords.append(kw)
        
        # 시/군/구 우선 할당 (예: "수원시"가 있으면 sigungu에, "장안구"도 있으면 detail_loc에)
        # 이 부분은 더 정교한 로직이 필요할 수 있음 (주소 파싱 라이브러리 등)
        if temp_keywords:
            # 가장 긴 키워드를 sigungu로, 그 다음을 detail_loc으로 할당하는 단순 로직 (개선 필요)
            sorted_kws = sorted(temp_keywords, key=len, reverse=True)
            if len(sorted_kws) > 0:
                # 시/군/구 이름으로 끝나는 경우가 많으므로, 그런 키워드를 우선적으로 sigungu에 할당
                sigungu_candidate = ""
                for kw_s in sorted_kws:
                    if kw_s.endswith("시") or kw_s.endswith("군") or kw_s.endswith("구"):
                        sigungu_candidate = kw_s
                        break
                if sigungu_candidate:
                    extracted_info["sigungu"] = sigungu_candidate
                elif sorted_kws[0] not in extracted_info["sido"]: # 시도와 중복되지 않게
                     extracted_info["sigungu"] = sorted_kws[0] # 임시로 가장 긴 것을 할당

                if len(sorted_kws) > 1:
                    if sorted_kws[1] not in extracted_info["sido"] and sorted_kws[1] != extracted_info["sigungu"]:
                        extracted_info["detail_loc"] = sorted_kws[1]
            extracted_info["keywords"].extend(temp_keywords)

        # 코스명 자체에서도 키워드 추출
        course_name_tokens = re.split(r'[\s\(\)\[\]]+', course_name_cleaned)
        for token in course_name_tokens:
            if any(city_keyword.lower() in token.lower() for city_keyword in list(KOREA_MAJOR_REGIONS.keys()) + SIGUNGU_AND_FEATURE_KEYWORDS):
                 if token not in extracted_info["keywords"]: extracted_info["keywords"].append(token)

        # 중복 제거 및 정제
        extracted_info["keywords"] = sorted(list(set(kw for kw in extracted_info["keywords"] if kw)))

        # 대표 주소 정보 설정
        addr_fields = [
            item_data.get("crsStAddr", ""), 
            item_data.get("crsEndAddr", ""), 
            item_data.get("sigunNm", ""), # 시군명칭 필드가 있다면 활용
            item_data.get("crsTourInfo", "")
        ]
        valid_addrs = [addr.strip() for addr in addr_fields if addr and addr.strip() and addr != "N/A"]
        if valid_addrs:
            # 가장 구체적이거나 긴 정보를 대표 주소로 (단순 길이 비교)
            extracted_info["best_address"] = max(valid_addrs, key=len)
        elif course_name_cleaned:
             extracted_info["best_address"] = course_name_cleaned # 주소 정보가 전혀 없으면 코스명이라도
        
        return extracted_info
    # --- 헬퍼 함수 끝 ---

    # 1) 관리용 루트 목록 수집
    routes = fetch_routes(num_of_rows=2000) 
    print(f"[PROCESS] 관리 루트 수집 완료: {len(routes)}건")
    if not routes:
        print("[Error] 관리 루트 수집에 실패했습니다.")
        return

    # 2) 각 루트별 실제 코스 수집
    course_items = []
    # ... (이전과 동일한 루트/코스 수집 로직) ...
    for r in routes:
        rid = r.get("routeIdx")
        if not rid: continue
        print(f"  ▶ 루트 {rid} ({r.get('themeNm', r.get('crsKorNm', ''))}) 코스 수집 중...")
        segs = fetch_courses(route_idx=rid, num_of_rows=500) 
        for c in segs:
            c["_routeIdx"] = rid
        course_items.extend(segs)
    print(f"[PROCESS] 전체 코스(segments) 수집 완료: {len(course_items)}건")
    if not course_items:
        raise RuntimeError("코스 아이템 수집 실패")

    # 3) 클리닝, 메타 준비, 임베딩용 텍스트 생성
    texts, ids, course_meta = [], [], {}
    for item in course_items:
        cleaned = clean_course_item(item) 
        if not cleaned.get("course_idx") or not cleaned.get("route_idx"): continue
        cid = f"{cleaned['route_idx']}_{cleaned['course_idx']}"
        
        course_name_cleaned = cleaned.get("course_name", "이름 없음")
        
        # ✅ 지역 정보 및 특징 키워드 추출
        extracted_details = extract_info_for_embedding(item, course_name_cleaned)
        
        try:
            distance_km = float(cleaned.get('distance', 0))
            minutes = (distance_km / 5) * 60 if distance_km > 0 else 0.0
        except (TypeError, ValueError):
            minutes = 0.0; distance_km = 0.0

        course_meta[cid] = {
            "course_name":    course_name_cleaned,
            "level":          cleaned.get("level", "정보 없음"),
            "distance":       distance_km,
            "minutes":        minutes,
            "summary":        cleaned.get("summary", ""),
            "contents":       cleaned.get("contents", ""),
            # ✅ 추출된 상세 지역 정보 메타에 저장
            "sido":           extracted_details["sido"],
            "sigungu":        extracted_details["sigungu"],
            "detail_loc":     extracted_details["detail_loc"],
            "loc_keywords":   extracted_details["keywords"], # 지역/특징 키워드
            "region_address": extracted_details["best_address"], # 가장 대표적인 주소/정보
            "start_address":  item.get("crsStAddr", "").strip(),
            "end_address":    item.get("crsEndAddr", "").strip(),
        }
        
        # ✅ 임베딩할 텍스트에 추출된 지역 정보와 특징 명시적으로 포함
        # 코스 유형 (예: 산책로, 등산로, 해안길 등)도 키워드로 추가하면 좋음
        # 현재는 extracted_details["keywords"]에 지역명 위주로 들어감.
        # 코스 설명에서 활동 유형(걷기, 러닝, 하이킹) 키워드도 추출하여 추가 가능.
        
        feature_keywords_text = ' '.join(extracted_details["keywords"])
        
        text_for_embedding = (
            f"코스 이름: {course_meta[cid]['course_name']}. "
            f"위치 정보: {course_meta[cid]['sido']} {course_meta[cid]['sigungu']} {course_meta[cid]['detail_loc']} {feature_keywords_text}. "
            f"대표 주소 또는 설명: {course_meta[cid]['region_address']}. "
            f"출발 지점: {course_meta[cid]['start_address']}. 도착 지점: {course_meta[cid]['end_address']}. "
            f"코스 요약: {course_meta[cid]['summary']}. "
            f"코스 상세 설명: {course_meta[cid]['contents']}. "
            f"난이도: {course_meta[cid]['level']}. 거리: {course_meta[cid]['distance']}km."
        )
        texts.append(text_for_embedding)
        ids.append(cid)

    if not texts:
        raise RuntimeError("임베딩할 텍스트 데이터가 없습니다.")

    # 4) 임베딩 생성 & FAISS 업서트
    embeddings = embed_texts(texts)
    print(f"[PROCESS] 임베딩 생성 완료 (개수: {len(embeddings)})")
    
    embedding_dimension = len(embeddings[0]) if embeddings else 1536 
    store = FaissStore(dim=embedding_dimension, index_path=str(idx_path), id_map_path=str(idmap_path))
    store.upsert(ids, embeddings)
    print("[PROCESS] FAISS 인덱스 저장 완료")

    # 5) 메타 저장
    with open(meta_file_path, "wb") as mf:
        pickle.dump(course_meta, mf)
    print(f"[PROCESS] 메타({meta_file_path.name}) 저장 완료")
    print("[PROCESS] 전체 인덱스 빌드 완료\n")

# 이하 main 함수는 이전과 동일합니다.
def main():
    idx_path   = base_dir / "faiss_index_all.pkl"
    idmap_path = base_dir / "id_map_all.pkl"
    meta_path  = base_dir / "routes_meta.pkl"
    src_files = [ base_dir / "main.py" ]
    
    rebuild_needed = False
    if not idx_path.exists() or not idmap_path.exists() or not meta_path.exists():
        rebuild_needed = True
        print("[INFO] 인덱스 또는 메타 파일이 존재하지 않아 새로 빌드합니다.")
    else:
        last_build_time = idx_path.stat().st_mtime
        for sf_path in src_files:
            if sf_path.exists() and sf_path.stat().st_mtime > last_build_time:
                rebuild_needed = True
                print(f"[INFO] 소스 파일 변경 감지 ({sf_path.name}): 기존 인덱스 및 메타 파일 삭제 후 리빌드합니다.")
                break
    
    if rebuild_needed:
        for f_path in (idx_path, idmap_path, meta_path):
            if f_path.exists():
                f_path.unlink()
        build_full_index()
    else:
        print("[SKIP] 전체 인덱스가 이미 존재하며 최신입니다.")

    required_keys = [
        "OPENAI_API_KEY", "DURUNUBI_API_KEY", "BIGDATA_API_KEY", "GEMINI_API_KEY"
    ]
    for key in required_keys:
        if not os.getenv(key):
            raise EnvironmentError(f"{key}가 설정되지 않았습니다. .env 파일을 확인하세요.")

    print("\n코스 조건 입력 (예: 부산 근처 가벼운 산책로, 서울 북쪽 하이킹 코스 등)")
    query = input("▶ 조건 입력: ").strip()

    region_name_for_pipeline = ""
    try:
        temp_openai_client = OpenAI()
        extraction_prompt = f"""사용자가 다음 질문에서 언급한 장소 또는 지역명이 있다면, 이 장소가 속하는 대한민국의 대표적인 '광역자치단체 축약 이름' (예: 서울, 부산, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주, 인천, 대전, 대구, 울산, 세종)을 하나만 정확히 추출해줘. 만약 여러 광역자치단체가 언급되거나 특정하기 어려우면 "전국"이라고 답변해줘. 다른 설명 없이 광역자치단체 축약 이름이나 "전국"만 답변해줘.
사용자 질문: "{query}"
답변:"""
        response = temp_openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=10, temperature=0
        )
        extracted_region = response.choices[0].message.content.strip()
        valid_region_keys = list(CITY_TO_AREACD_FOR_PREDICTION.keys()) + ["전국"]
        if extracted_region in valid_region_keys:
            if extracted_region == "전국":
                print("[INFO] LLM: 질문에서 특정 지역을 파악하기 어렵습니다. 전국 대상으로 검색합니다.")
                region_name_for_pipeline = ""
            else:
                region_name_for_pipeline = extracted_region
                print(f"[INFO] LLM 추출 대표 지역명 (pipeline 전달용): {region_name_for_pipeline}")
        else:
            print(f"[INFO] LLM이 지역명을 추출했으나({extracted_region}), 시스템 정의 지역명과 다릅니다. 전국 대상으로 검색합니다.")
            region_name_for_pipeline = ""
    except Exception as e:
        print(f"[Error] LLM으로 지역명 추출 중 오류 발생: {e}. 폴백 로직을 사용합니다.")
        for r_name in CITY_TO_AREACD_FOR_PREDICTION.keys():
            if r_name in query:
                region_name_for_pipeline = r_name
                break
        if region_name_for_pipeline: print(f"[INFO] 폴백 추출 대표 지역명: {region_name_for_pipeline}")
        else: print("[INFO] 질문에 특정 지역명이 명시되지 않았습니다 (폴백).")

    store = FaissStore(dim=1536, index_path=str(idx_path), id_map_path=str(idmap_path))

    today_dt = datetime.today()
    default_date_str = today_dt.strftime("%Y%m%d")
    days_limit = 7
    latest_forecast_date_dt = today_dt + timedelta(days=days_limit)
    latest_forecast_date_str = latest_forecast_date_dt.strftime("%Y%m%d")
    print(f"\n[INFO] 날씨 예보를 포함한 추천은 오늘({default_date_str})부터 일주일({latest_forecast_date_str})까지만 가능합니다.")
    while True:
        date_str = input(f"▶ 추천 받고 싶은 날짜를 입력해주세요 (YYYYMMDD) [기본값: 오늘 - {default_date_str}]: ").strip() or default_date_str
        try:
            target_date_dt = datetime.strptime(date_str, "%Y%m%d")
            if not (today_dt.date() <= target_date_dt.date() <= latest_forecast_date_dt.date()):
                print(f"[주의] 입력하신 날짜({date_str})는 예보 가능 범위({default_date_str}~{latest_forecast_date_str})를 벗어났습니다. 다시 입력해주세요.")
                continue
            break
        except ValueError:
            print("[오류] 날짜 형식이 잘못되었습니다. YYYYMMDD 형식으로 입력해주세요.")

    print("\n[PROCESS] 추천 생성 중...")
    pipeline = RAGPipeline(
        store=store, fetch_k=20, suggest_k=2, region_name=region_name_for_pipeline
    )
    result_text = pipeline.run(query, date_str)

    print("\n==================== 추천 코스 ====================")
    print(result_text)
    print("===================================================\n")

def main_pipeline(query: str, date_str: str = None):
    """
    외부(Flask 등)에서 바로 호출 가능한 추천 파이프라인 함수.
    """
    idx_path   = base_dir / "faiss_index_all.pkl"
    idmap_path = base_dir / "id_map_all.pkl"

    store = FaissStore(dim=1536, index_path=str(idx_path), id_map_path=str(idmap_path))

    if not date_str:
        date_str = datetime.today().strftime("%Y%m%d")

    # 지역명 추출 부분을 간단히 처리 (실 서비스에선 필요시 LLM/정규식 추출)
    region_name_for_pipeline = ""
    for r_name in CITY_TO_AREACD_FOR_PREDICTION.keys():
        if r_name in query:
            region_name_for_pipeline = r_name
            break

    pipeline = RAGPipeline(
        store=store,
        fetch_k=20,
        suggest_k=2,
        region_name=region_name_for_pipeline
    )
    # 결과만 string으로 출력
    result, _, _ = pipeline.run(query, date_str)
    return result

if __name__ == "__main__":
    main()


