#!/usr/bin/env python3
# programming/module/rag/pipeline.py

import os
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI

# 모듈 임포트
from module.embedding.embedder import embed_text
from module.vector_store.faiss_store import FaissStore
from module.models.forecast_models import predict_visitor_stats
from module.api_clients.weather_gemini import get_weather_forecast_from_gemini

# 지역명 → areaCd 매핑 테이블 (DataLab API 예측용)
CITY_TO_AREACD_FOR_PREDICTION = {
    "서울": "11", "부산": "26", "대구": "27", "인천": "28",
    "광주": "29", "대전": "30", "울산": "31", "세종": "36",
    "경기": "31", "강원": "32", "충북": "33", "충남": "34",
    "전북": "35", "전남": "36", "경북": "37", "경남": "38", "제주": "39",
    "서울특별시": "11", "부산광역시": "26", "대구광역시": "27", "인천광역시": "28",
    "광주광역시": "29", "대전광역시": "30", "울산광역시": "31", "세종특별자치시": "36",
    "경기도": "31", "강원특별자치도": "32", "충청북도": "33", "충청남도": "34",
    "전북특별자치도": "35", "전라남도": "36", "경상북도": "37", "경상남도": "38", "제주특별자치도": "39"
}

def parse_area_cd_for_prediction(query_or_region_name: str) -> str:
    if query_or_region_name in CITY_TO_AREACD_FOR_PREDICTION:
        return CITY_TO_AREACD_FOR_PREDICTION[query_or_region_name]
    for city_keyword, code in CITY_TO_AREACD_FOR_PREDICTION.items():
        if city_keyword in query_or_region_name:
            return code
    return ""

# OpenAI 클라이언트 설정
# .env 파일 로드 (프로젝트 루트에 .env가 있다고 가정. evaluation.py와 동일한 방식)
current_pipeline_script_file = Path(__file__).resolve()
pipeline_project_root = current_pipeline_script_file.parents[2] # programming/
dotenv_path_pipeline = pipeline_project_root / ".env"
if dotenv_path_pipeline.exists():
    load_dotenv(dotenv_path=dotenv_path_pipeline)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # API 키가 없으면 LLM 답변 생성 부분에서 오류 발생 가능성 있음
    print("[WARNING] pipeline.py: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다. LLM 답변 생성이 제한될 수 있습니다.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)


def get_congestion_level(total_visitors: Optional[float]) -> str:
    if total_visitors is None: return "예측 정보 없음"
    if total_visitors > 5_000_000: return "매우 붐빌 것으로 예상돼요. 참고하세요."
    elif total_visitors > 3_000_000: return "다소 활기찰 것으로 보여요."
    elif total_visitors >= 0:
        return "비교적 한산할 것으로 예상돼요."
    return "방문객 수 예측 데이터를 해석할 수 없습니다."

class RAGPipeline:
    def __init__(
        self,
        store: FaissStore,
        fetch_k: int = 50, # 이전 실험에서 50으로 설정
        suggest_k: int = 2, # LLM에게 최종 추천하도록 요청할 개수는 유지 (보통 2-3개)
        region_name: str = ""
    ):
        self.store = store
        self.fetch_k = fetch_k
        self.suggest_k = suggest_k
        self.region_name_from_main = region_name.strip()
        # OpenAI 클라이언트는 클래스 외부에서 이미 초기화됨
        self.openai_client = openai_client


    def _extract_location_keywords_from_query(self, query: str) -> List[str]:
        known_locations = list(CITY_TO_AREACD_FOR_PREDICTION.keys()) + \
                          [ "강남", "해운대", "수원", "장안구", "충주", "충주시",
                           "종로", "중구", "서귀포시"
                          ] 
        found_keywords = []
        query_tokens = re.split(r'[\s,]+', query)
        for token in query_tokens:
            for loc in known_locations:
                if loc in token or token in loc:
                    if loc not in found_keywords:
                         found_keywords.append(loc)
        if self.region_name_from_main and self.region_name_from_main not in found_keywords:
            found_keywords.append(self.region_name_from_main)
        print(f"[DEBUG pipeline] 추출된 지역 검색 키워드: {found_keywords}")
        return found_keywords

    def run(self, query: str, target_date: str) -> tuple[str, list, list]: # 반환 타입 유지
        print(f"[DEBUG pipeline.run] 시작. query='{query}', target_date='{target_date}', initial_region_name='{self.region_name_from_main}'")

        # 1) 메타 로드
        try:
            # pipeline.py는 programming/module/rag/에 위치
            # routes_meta.pkl은 programming/에 위치
            meta_file_path = pipeline_project_root / "routes_meta.pkl"
            print(f"[DEBUG pipeline.run] 메타 파일 경로: {meta_file_path}")
            with open(meta_file_path, "rb") as mf:
                route_meta = pickle.load(mf)
            print(f"[DEBUG pipeline.run] 메타 로드 완료: {len(route_meta)}개 코스 메타")
        except FileNotFoundError:
            return "오류: 코스 메타 정보를 불러올 수 없습니다.", [], []
        except Exception as e:
            return f"오류: 코스 메타 정보 처리 중 문제 발생 ({e})", [], []

        # 2) 임베딩 검색 (1차 후보군 확보)
        q_vec = embed_text(query, model="text-embedding-3-small")
        initial_candidates_with_scores = self.store.search(q_vec, top_k=self.fetch_k)
        
        if not initial_candidates_with_scores:
            return "죄송합니다, 현재 조건과 유사한 코스를 찾지 못했습니다.", [], []
        
        print(f"[DEBUG pipeline.run] FAISS 초기 검색 결과: {len(initial_candidates_with_scores)}건")

        # ✅ [수정됨] 평가용 retrieved_ids는 초기 FAISS 검색 결과에서 바로 생성
        retrieved_ids_for_evaluation = [cid for cid, score in initial_candidates_with_scores]
        print(f"[DEBUG pipeline.run] 평가용 Retrieved IDs (FAISS 원본 순서): {retrieved_ids_for_evaluation[:10]}...")


        # 3) 사용자 쿼리에서 지역 키워드 추출 (기존 로직 유지)
        location_keywords_from_query = self._extract_location_keywords_from_query(query)

        # 3-1) 지역 연관성 기반 재정렬 로직 (LLM에게 전달할 후보 선정을 위해 기존 로직 유지)
        # 이 부분은 최종 답변 생성에는 사용되지만, 검색 성능 평가에는 사용되지 않음
        highly_relevant_courses = []
        other_courses = []
        # ... (사용자님의 기존 재정렬 로직: initial_candidates_with_scores를 순회하며 meta 정보와 location_keywords_from_query 비교)
        # 이 로직은 사용자님이 올려주신 pipeline.py의 코드를 그대로 사용한다고 가정합니다.
        # (아래는 사용자님이 올려주신 코드의 재정렬 로직 요약입니다)
        if not location_keywords_from_query and not self.region_name_from_main:
            print("[INFO pipeline.run] 쿼리에 지역 언급이 없어 FAISS 순위대로 LLM 후보 구성합니다.")
            for cid, initial_score in initial_candidates_with_scores: # 초기 FAISS 결과를 사용
                meta = route_meta.get(cid, {})
                if meta:
                    # other_courses에 추가 (LLM 후보군으로 바로 사용 가능)
                    # 이 예시에서는 other_courses가 final_candidate_pool 역할을 함
                    other_courses.append({'cid': cid, 'score': initial_score, 'meta': meta, 'relevance_level': 0, 'matched_kws': []})
            final_candidate_pool_for_llm = other_courses # final_candidate_pool_for_llm로 명칭 변경
        else:
            for cid, initial_score in initial_candidates_with_scores: # 초기 FAISS 결과를 사용
                meta = route_meta.get(cid, {})
                if not meta: continue
                
                text_to_check_relevance = (
                    f"{meta.get('course_name','')}"
                    f" {meta.get('region_address','')}" 
                    f" {meta.get('summary','')}"
                    f" {meta.get('start_address','')}"
                    f" {meta.get('end_address','')}"
                ).lower()

                specific_keyword_match_count = 0
                matched_kws_for_course = []
                if location_keywords_from_query:
                    for kw in location_keywords_from_query:
                        if kw.lower() in text_to_check_relevance:
                            specific_keyword_match_count += 1
                            if kw not in matched_kws_for_course: matched_kws_for_course.append(kw)
                
                main_region_match = False
                if self.region_name_from_main and self.region_name_from_main.lower() in text_to_check_relevance:
                    main_region_match = True
                    if self.region_name_from_main not in matched_kws_for_course: matched_kws_for_course.append(self.region_name_from_main)

                relevance_level = 0
                if specific_keyword_match_count == len(location_keywords_from_query) and location_keywords_from_query:
                    relevance_level = 3
                elif specific_keyword_match_count > 0:
                    relevance_level = 2
                elif main_region_match:
                    relevance_level = 1
                
                if relevance_level >= 2: 
                    highly_relevant_courses.append({'cid': cid, 'score': initial_score, 'meta': meta, 'relevance_level': relevance_level, 'matched_kws': matched_kws_for_course})
                else:
                    other_courses.append({'cid': cid, 'score': initial_score, 'meta': meta, 'relevance_level': relevance_level, 'matched_kws': matched_kws_for_course})

            highly_relevant_courses.sort(key=lambda x: (-x['relevance_level'], -x['score']))
            other_courses.sort(key=lambda x: (-x['relevance_level'], -x['score']))
            final_candidate_pool_for_llm = highly_relevant_courses + other_courses # LLM 전달용 후보군
        
        print(f"\n[DEBUG pipeline.run] --- 지역 연관성 우선 정렬 후 LLM 전달용 상위 5개 후보 ---")
        for i, cand_info in enumerate(final_candidate_pool_for_llm[:5]):
            meta = cand_info['meta']
            print(f"  {i+1}. CID: {cand_info['cid']}, FAISS Score: {cand_info['score']:.4f}, Relevance: {cand_info['relevance_level']}, Matched: {cand_info.get('matched_kws')}")
            print(f"     Name: {meta.get('course_name', 'N/A')}")

        # 4) 최종 후보(suggest_k 개)에 대한 설명 준비 (LLM 프롬프트용)
        texts_for_llm: List[str] = []
        # LLM에게는 재정렬된 후보 중 상위 self.suggest_k개만 전달
        for cand_info in final_candidate_pool_for_llm[:self.suggest_k]: 
            meta = cand_info['meta']
            # ... (이전의 desc_for_prompt 생성 로직과 유사하게, relevance_text 생성도 여기에 포함) ...
            # 이 부분은 사용자님이 올려주신 pipeline.py의 로직을 따른다고 가정합니다.
            # (아래는 사용자님이 올려주신 코드의 texts_for_llm 생성 로직 요약입니다)
            relevance_text = "지역 연관성: 정보 없음"
            if cand_info.get('relevance_level', 0) >= 2 :
                 relevance_text = f"지역 연관성: 사용자 요청 지역 키워드({', '.join(cand_info.get('matched_kws',[]))})와(과) 관련성이 높습니다."
            elif cand_info.get('relevance_level', 0) == 1:
                 relevance_text = f"지역 연관성: 사용자 주요 언급 지역({self.region_name_from_main if self.region_name_from_main else ', '.join(cand_info.get('matched_kws',[]))})과(과) 관련이 있습니다."
            elif location_keywords_from_query or self.region_name_from_main:
                 relevance_text = f"지역 연관성: 사용자 요청 지역과 직접적인 관련성은 낮을 수 있습니다."
            desc_for_prompt = (
                f"코스명: {meta.get('course_name', '이름 없음')}\n"
                f"요약: {meta.get('summary', '요약 정보 없음')}\n"
                f"{relevance_text}"
            ) # 실제로는 더 많은 코스 정보를 포함할 것입니다.
            texts_for_llm.append(desc_for_prompt)


        if not texts_for_llm: # LLM에게 전달할 후보가 없으면
             # 이 경우 평가용 ID는 있을 수 있으나, LLM 답변은 생성 불가
            return "죄송합니다, 최종 추천 드릴 코스 정보를 구성하지 못했습니다.", retrieved_ids_for_evaluation, []
        
        print(f"[DEBUG pipeline.run] LLM에 전달할 후보 코스 정보 생성 완료 (상위 {len(texts_for_llm)}개)")

        # 5) 방문객 예측을 위한 지역 코드 결정 (기존 로직 유지)
        prediction_area_cd_for_stats = parse_area_cd_for_prediction(self.region_name_from_main or query)
        
        # 6) 방문객 수 예측 및 혼잡도 텍스트 생성 (기존 로직 유지 또는 임시 비활성화된 상태로 유지)
        # ... (사용자님의 기존 코드 또는 임시 비활성화된 코드) ...
        # (아래는 사용자님이 올려주신 코드의 방문객 예측 로직 요약입니다)
        congestion_guidance = "혼잡도 예측 정보를 사용할 수 없습니다."
        if prediction_area_cd_for_stats:
            try:
                predicted_stats_data = predict_visitor_stats(area_cd=prediction_area_cd_for_stats, target_date_str=target_date)
                if predicted_stats_data:
                    total_visitors = predicted_stats_data.get('predicted_local_visitors', 0) + predicted_stats_data.get('predicted_non_local_visitors', 0)
                    congestion_guidance = get_congestion_level(total_visitors)
                else:
                    congestion_guidance = f"해당 지역({self.region_name_from_main or prediction_area_cd_for_stats})의 방문객 수 예측 데이터를 가져올 수 없어, 혼잡도 파악이 어렵습니다."
            except Exception as e: # API 트래픽 초과 등 예외 처리
                print(f"[PIPELINE WARNING] 방문객 수 예측 중 오류 발생: {e}")
                congestion_guidance = "방문객 수 예측 중 오류가 발생하여 혼잡도 정보를 제공할 수 없습니다."
        else:
            congestion_guidance = "질문에서 지역 정보를 명확히 파악하기 어려워 방문객 수 예측 및 혼잡도 파악을 수행할 수 없습니다."

        # 7) 날씨 예보 조회 (Gemini API 사용 - 기존 로직 유지)
        # ... (사용자님의 기존 코드) ...
        # (아래는 사용자님이 올려주신 코드의 날씨 예보 로직 요약입니다)
        weather_info = f"{target_date}의 날씨 예보를 가져올 수 없습니다."
        location_for_gemini = self.region_name_from_main 
        if not location_for_gemini: 
            if location_keywords_from_query: 
                location_for_gemini = location_keywords_from_query[0] 
        if location_for_gemini:
            gemini_weather_data = get_weather_forecast_from_gemini(location=location_for_gemini, target_date_str=target_date)
            if gemini_weather_data:
                try:
                    overall = gemini_weather_data.get("overall_weather", "정보 없음")
                    min_temp = gemini_weather_data.get("min_temp_celsius", "N/A")
                    max_temp = gemini_weather_data.get("max_temp_celsius", "N/A")
                    precip_prob = gemini_weather_data.get("precipitation_probability_percent")
                    comment = gemini_weather_data.get("brief_comment", "")
                    lines = [f"- 날씨: {overall}", f"- 예상 기온: 최저 {min_temp}°C / 최고 {max_temp}°C"]
                    if precip_prob is not None and str(precip_prob).lower() != "null": lines.append(f"- 강수 확률: {precip_prob}%")
                    weather_info = f"{target_date} ({location_for_gemini}) 예상 날씨:\n" + "\n".join(lines)
                    if comment and str(comment).lower() != "null": weather_info += f"\n- 코멘트: {comment}"
                except Exception as e:
                    weather_info = f"{target_date} 날씨 정보를 처리하는 중 문제가 발생했습니다 ({e})."
            else:
                weather_info = f"{target_date} ({location_for_gemini}) 날씨 예보를 Gemini API에서 가져오지 못했습니다."
        else:
            weather_info = f"질문에서 지역 정보가 명확하지 않아 {target_date} 날씨 예보를 조회할 수 없습니다."

        # 8) LLM 프롬프트 구성 (아래 부분 보강)
        numbered_course_info_for_llm = "\n\n".join(
            f"후보 {i+1}:\n{texts_for_llm[i]}" for i in range(len(texts_for_llm))
        )
        prompt = (
            "당신은 사용자에게 맞춤형 코스 정보를 상세하고 친절하게 설명하는 여행 전문가입니다. "
            "단, 프로그램에서 추천해줄 수 없는 부분에 관한 이야기는 하지 마세요.\n"
            "다음은 사용자의 질문과 관련된 후보 코스 정보, 각 코스와 사용자 질문의 지역적 연관성, 그리고 선택한 날짜의 기상 정보입니다.\n\n"
            "--- 후보 코스 정보 (검색된 순서대로) ---\n"
            f"{numbered_course_info_for_llm}\n\n"
            "--- 예상 방문객 혼잡도 ---\n"
            f"{congestion_guidance}\n\n"
            "--- 선택 날짜 기상 정보 ---\n"
            f"{weather_info}\n\n"
            "--- 요청사항 ---\n"
            "1. 반드시 위의 '선택 날짜 기상 정보'도 사용자에게 안내해주세요.\n"
            "2. 우리 프로그램에서 지원하지 않는 항목(예: 교통 등)은 안내하지 마세요.\n"
            f"3. 위 '후보 코스 정보' 중에서, 사용자 질문(\"{query}\")에 가장 적합하다고 판단되는 코스를 **{self.suggest_k}개** 선정해 주세요.\n"
            "4. 각 코스에 대해 이름/위치/거리/특징을 구체적으로 설명하고, 코스 선택 이유를 알려주세요. 개조식으로 설명하지 말고 서술식으로 설명하세요.\n"
            "5. 가능한 한 정확하게 안내하세요.\n"
            "6. 안내에 포함되지 않은 부분(교통 등)은 언급하지 마세요."
            "7. 이모지 등을 활용해 친절하게 설명하세요"
        )

        # 9) LLM 호출 및 결과 반환
        final_answer_text = "LLM 호출에 실패했거나 OpenAI 클라이언트가 설정되지 않았습니다."
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "당신은 한국의 걷기·러닝·하이킹 코스 전문가입니다. 주어진 후보 리스트와 예측·기상 정보를 바탕으로, 사용자 질문에 맞는 최적의 코스를 추천합니다."},
                        {"role": "user", "content": prompt}
                    ]
                )
                final_answer_text = response.choices[0].message.content
            except Exception as e:
                print(f"[ERROR pipeline.run] OpenAI API 호출 중 오류: {e}")
                final_answer_text = "추천 코스 생성 중 오류가 발생했습니다."

        # 평가용 ID 목록과 LLM 전달용 컨텍스트 목록 반환
        retrieved_context_for_eval = texts_for_llm
        return final_answer_text, retrieved_ids_for_evaluation, retrieved_context_for_eval

if __name__ == '__main__':
    pass
