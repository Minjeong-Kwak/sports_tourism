# programming/module/embedding/embedder.py

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# 1) .env에서 키 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

# 2) 클라이언트 초기화
client = OpenAI(api_key=api_key)

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    여러 텍스트를 배치로 받아 임베딩 벡터를 반환
    """
    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    return [d.embedding for d in resp.data]

def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    단일 텍스트 임베딩
    """
    return embed_texts([text], model=model)[0]
