import os
import faiss
import numpy as np
import pickle
from typing import List, Tuple

class FaissStore:
    def __init__(self, dim: int, index_path: str, id_map_path: str):
        """
        :param dim: 임베딩 벡터 차원
        :param index_path: FAISS 인덱스 파일 경로
        :param id_map_path: id→인덱스 매핑 파일 경로
        """
        self.dim = dim
        self.index_path = index_path
        self.id_map_path = id_map_path

        # 1) 인덱스와 id_map 로드
        self.index = self._load_index()
        self.id_map = self._load_id_map()
        # 2) 역매핑 테이블 생성
        self.rev_map = {v: k for k, v in self.id_map.items()}

    def _load_index(self) -> faiss.Index:
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        # Cosine 유사도를 위해 내부적으로 IP를 쓰면서 벡터를 정규화
        return faiss.IndexFlatIP(self.dim)

    def _save_index(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

    def _load_id_map(self) -> dict:
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_id_map(self) -> None:
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        """
        :param ids: 각 벡터의 고유 식별자 리스트
        :param vectors: 임베딩 벡터 리스트
        """
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        # 인덱스에 추가
        self.index.add(arr)

        # id_map 및 rev_map 갱신
        base = len(self.id_map)
        for i, doc_id in enumerate(ids):
            idx = base + i
            self.id_map[doc_id] = idx
            self.rev_map[idx] = doc_id

        # 저장
        self._save_index()
        self._save_id_map()

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        :param query_vec: 질의 임베딩 벡터
        :param top_k: 상위 k개 결과
        :return: [(id, 유사도 점수), ...]
        """
        q = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            # -1 인덱스 스킵
            if idx < 0:
                continue
            # 역매핑으로 id 조회
            doc_id = self.rev_map.get(int(idx))
            if doc_id is None:
                continue
            results.append((doc_id, float(score)))

        return results
