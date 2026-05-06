from pydantic import BaseModel, Field, field_validator
from config import CRITIQUE_PASS_SCORE


class RoutingResult(BaseModel):
    bot_id: str
    name: str
    similarity_score: float = Field(ge=0.0, le=1.01)  # 1.01 tolerates float precision from FAISS IndexFlatIP


class BotPost(BaseModel):
    bot_id: str
    topic: str
    post_content: str = Field(max_length=280)
    retries: int = Field(default=0, ge=0)


class PostScore(BaseModel):
    score: int = Field(ge=1, le=10)
    feedback: str

    @property
    def passed(self) -> bool:
        return self.score >= CRITIQUE_PASS_SCORE


class DefenseReply(BaseModel):
    reply: str
    injection_detected: bool = False


class LLMCallStats(BaseModel):
    phase: str
    latency_ms: int
    estimated_tokens: int
