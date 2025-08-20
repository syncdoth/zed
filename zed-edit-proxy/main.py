#!/usr/bin/env python3
"""
Zed Edit Prediction Proxy

A proxy service that adapts Zed's edit prediction API format to OpenAI-compatible APIs,
allowing you to use local models with vLLM, Ollama, or other OpenAI-compatible servers.

Usage:
    python main.py --config config.yaml

Environment Variables:
    ZED_EDIT_PROXY_PORT: Port to run the proxy on (default: 8080)
    ZED_EDIT_PROXY_HOST: Host to bind to (default: 127.0.0.1)
    ZED_EDIT_PROXY_DEBUG: Enable debug logging (default: false)
"""

import json
import logging
import uuid
from typing import Optional
import argparse

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from httpx import AsyncClient, RequestError, HTTPStatusError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Zed Edit Prediction Proxy", version="1.0.0")


class Config:
    """Configuration for the proxy service"""

    def __init__(self, config_path: Optional[str] = None):
        self.openai_base_url = "http://localhost:1234/v1"  # Default for LM Studio
        self.model_name = "zed-industries/zeta"
        self.max_tokens = 2048
        self.temperature = 0.1
        self.timeout = 30.0
        self.system_prompt = self._default_system_prompt()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            self.openai_base_url = config.get("openai_base_url", self.openai_base_url)
            self.model_name = config.get("model_name", self.model_name)
            self.max_tokens = config.get("max_tokens", self.max_tokens)
            self.temperature = config.get("temperature", self.temperature)
            self.timeout = config.get("timeout", self.timeout)
            self.system_prompt = config.get("system_prompt", self.system_prompt)

            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    def _default_system_prompt(self) -> str:
        return """You are Zeta, an AI assistant that predicts code edits based on user input and context.

Your task is to analyze the provided code context and predict what edits the user likely wants to make. Focus on:
1. Understanding the current code structure and context
2. Predicting logical next steps or completions
3. Maintaining code style and conventions
4. Being concise and accurate

Respond only with the predicted code changes, not explanations or commentary."""


# Zed API Models
class PredictEditsGitInfo(BaseModel):
    head_sha: Optional[str] = None
    remote_origin_url: Optional[str] = None
    remote_upstream_url: Optional[str] = None


class PredictEditsBody(BaseModel):
    outline: Optional[str] = None
    input_events: str
    input_excerpt: str
    speculated_output: Optional[str] = None
    can_collect_data: bool = False
    diagnostic_groups: Optional[list] = None
    git_info: Optional[PredictEditsGitInfo] = None


class PredictEditsResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    output_excerpt: str


class AcceptEditPredictionBody(BaseModel):
    request_id: str


# OpenAI API Models
class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


class OpenAIChoice(BaseModel):
    message: OpenAIMessage
    finish_reason: Optional[str] = None


class OpenAIResponse(BaseModel):
    choices: list[OpenAIChoice]


# Global configuration
config = Config()


class RequestTransformer:
    """Transforms Zed edit prediction requests to OpenAI format"""

    @staticmethod
    def zed_to_openai(zed_request: PredictEditsBody) -> OpenAIRequest:
        """Transform Zed's PredictEditsBody to OpenAI chat completion format"""

        # Build context from the request
        context_parts = []

        # Add outline if provided
        if zed_request.outline:
            context_parts.append(f"Code outline:\n{zed_request.outline}")

        # Add the current code context
        context_parts.append(f"Current code context:\n{zed_request.input_excerpt}")

        # Add input events (user actions)
        if zed_request.input_events:
            context_parts.append(f"User input events:\n{zed_request.input_events}")

        # Add speculated output if available
        if zed_request.speculated_output:
            context_parts.append(f"Speculated output:\n{zed_request.speculated_output}")

        # Add diagnostic information if available
        if zed_request.diagnostic_groups:
            diagnostics = []
            for group_name, group_data in zed_request.diagnostic_groups:
                diagnostics.append(f"{group_name}: {json.dumps(group_data)}")
            context_parts.append("Diagnostics:\n" + "\n".join(diagnostics))

        # Add git context if available and data collection is allowed
        if zed_request.can_collect_data and zed_request.git_info:
            git_info = []
            if zed_request.git_info.head_sha:
                git_info.append(f"HEAD SHA: {zed_request.git_info.head_sha}")
            if zed_request.git_info.remote_origin_url:
                git_info.append(f"Origin: {zed_request.git_info.remote_origin_url}")
            if git_info:
                context_parts.append("Git context:\n" + "\n".join(git_info))

        # Combine all context
        full_context = "\n\n".join(context_parts)

        # Create the user message
        user_message = f"""Based on the following context, predict the most likely code edit or completion:

{full_context}

Respond with only the predicted code changes or completion."""

        messages = [
            OpenAIMessage(role="system", content=config.system_prompt),
            OpenAIMessage(role="user", content=user_message),
        ]

        return OpenAIRequest(
            model=config.model_name,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=False,
        )


class ResponseTransformer:
    """Transforms OpenAI responses to Zed format"""

    @staticmethod
    def openai_to_zed(openai_response: OpenAIResponse, request_id: Optional[str] = None) -> PredictEditsResponse:
        """Transform OpenAI response to Zed's PredictEditsResponse format"""

        if not openai_response.choices:
            raise ValueError("OpenAI response has no choices")

        # Get the first choice's message content
        output_excerpt = openai_response.choices[0].message.content

        # Clean up the output - remove any markdown formatting if present
        if output_excerpt.startswith("```") and output_excerpt.endswith("```"):
            lines = output_excerpt.split("\n")
            if len(lines) > 2:
                # Remove first and last lines (markdown delimiters)
                output_excerpt = "\n".join(lines[1:-1])

        return PredictEditsResponse(request_id=request_id or str(uuid.uuid4()), output_excerpt=output_excerpt.strip())


# HTTP Client for OpenAI API calls
http_client = AsyncClient()


@app.on_event("startup")
async def startup():
    logger.info("Starting Zed Edit Prediction Proxy")
    logger.info(f"OpenAI Base URL: {config.openai_base_url}")
    logger.info(f"Model: {config.model_name}")


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


@app.post("/predict_edits/v2")
async def predict_edits(request: PredictEditsBody) -> PredictEditsResponse:
    """
    Main endpoint that mimics Zed's predict_edits/v2 API
    """
    request_id = str(uuid.uuid4())

    logger.info(f"Processing edit prediction request {request_id}")
    logger.debug(f"Request: {request.model_dump()}")

    try:
        # Transform Zed request to OpenAI format
        openai_request = RequestTransformer.zed_to_openai(request)
        logger.debug(f"OpenAI request: {openai_request.model_dump()}")

        # Make request to OpenAI-compatible API
        response = await http_client.post(
            f"{config.openai_base_url}/chat/completions", json=openai_request.model_dump(), timeout=config.timeout
        )
        response.raise_for_status()

        # Parse OpenAI response
        openai_response_data = response.json()
        openai_response = OpenAIResponse(**openai_response_data)
        logger.debug(f"OpenAI response: {openai_response.model_dump()}")

        # Transform back to Zed format
        zed_response = ResponseTransformer.openai_to_zed(openai_response, request_id)

        logger.info(f"Successfully processed request {request_id}")
        return zed_response

    except HTTPStatusError as e:
        logger.error(f"HTTP error from OpenAI API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e.response.status_code}")
    except RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to OpenAI API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict_edits/accept")
async def accept_edit_prediction(request: AcceptEditPredictionBody):
    """
    Endpoint for accepting edit predictions - currently a no-op
    In a full implementation, this could log acceptance for analytics
    """
    logger.info(f"Edit prediction accepted: {request.request_id}")
    return {"status": "accepted"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection to OpenAI API
        response = await http_client.get(f"{config.openai_base_url}/models", timeout=5.0)
        api_healthy = response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        api_healthy = False

    return {
        "status": "healthy" if api_healthy else "degraded",
        "openai_api_connected": api_healthy,
        "config": {"openai_base_url": config.openai_base_url, "model_name": config.model_name},
    }


def main():
    parser = argparse.ArgumentParser(description="Zed Edit Prediction Proxy")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    global config
    config = Config(args.config)

    logger.info(f"Starting proxy on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if not args.debug else "debug")


if __name__ == "__main__":
    main()
