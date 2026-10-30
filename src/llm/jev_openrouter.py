from dotenv import load_dotenv

import os
from dotenv import load_dotenv
from typing import Dict, Any

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

class JevLoader:


    def __init__(self,
                model: str = "typesafe/jev-1.14",
                timeout_seconds: float = 30,
                max_retries: int = 3,
                retry_seconds: float = 60
    ):

        self._client = self._load_client()
        self._jev_model = model

        self._timeout = timeout_seconds
        self._retry_policy = RetryPolicy(
            max_retries=max_retries,
            timeout=retry_seconds,
        )


    # ====================== INITIALIZE =====================

    @staticmethod
    def _load_client():

        load_dotenv()
        return AsyncTypeSafeClient(api_key=os.getenv("OPENROUTER_API_KEY"),
                                   base_url="https://openrouter.ai/api")


    async def __aenter__(self) -> "JevLoader":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        close = getattr(self._client, "__aexit__", None)
        if close is not None:
            await close(None, None, None)


    # ========================== CALL MODEL ============================

    def _get_params(self):
        return {
            "model": self._jev_model,
            "retry": self._retry_policy,
            "timeout": self._timeout
        }


    # ========================== CALL MODEL ============================

    async def choice(self, state: str, instruction: str, criteria: Dict[str, Any]) -> str:

        results = await self._client.system_one(
            state=state,
            questions={
                "question_choice": {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": criteria
                }
            },
            **self._get_params()
        )

        return results.answers["question_choice"]


    async def noul(self, state: str, instruction: str) -> float:

        results = await self._client.system_one(
            state=state,
            questions={
                "question_noul": {
                    "type": "noul",
                    "instructions": instruction,
                }
            },
            **self._get_params()
        )
        
        return results.answers["question_noul"]


    async def jev_multi(self, state: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:

        results = await self._client.system_one(
            state=state,
            questions=questions,
            **self._get_params()
        )

        return results.answers
     