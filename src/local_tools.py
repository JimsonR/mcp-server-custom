from __future__ import annotations

import json
import os
import re
import requests
import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Session-scoped in-memory store.
# Persists for the lifetime of the Python process.
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}

# Default timeout and token settings for analyst agents
DEFAULT_TIMEOUT = 120  # Increased from 25 for reasoning models
DEFAULT_MAX_TOKENS = 4000  # Ensure enough tokens for full responses


def _get_openai_client(api_key: str) -> OpenAI:
    """Create OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _make_llm_call(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    agent_type: str = "unknown"
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Make LLM call using OpenAI SDK with proper fallback handling.
    
    Returns:
        Tuple of (content, usage_dict, error_message)
        - content: The extracted analysis content
        - usage_dict: Token usage information
        - error_message: None if successful, error string if failed
    """
    logger.info(f"Starting {agent_type} LLM call with model {model}")
    
    try:
        client = _get_openai_client(api_key)
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/xendex-mcp-server",
                "X-Title": f"Xendex MCP {agent_type.replace('_', ' ').title()}",
            },
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=timeout,
            max_tokens=max_tokens,
        )
        
        # Extract content from response
        choice = completion.choices[0] if completion.choices else None
        if not choice:
            logger.error(f"{agent_type}: No choices in response")
            return "", {}, f"[ERROR] {agent_type} agent received no response choices from model"
        
        message = choice.message
        raw_content = message.content or ""
        
        logger.info(f"{agent_type}: Raw content length: {len(raw_content)}")
        
        # OpenRouter may put reasoning in a separate field for DeepSeek R1
        # Check if content is empty but reasoning exists
        reasoning = getattr(message, 'reasoning', None) or ""
        if not raw_content and reasoning:
            logger.info(f"{agent_type}: Content empty, using reasoning field (length: {len(reasoning)})")
            raw_content = reasoning
        
        # Extract actual content from response (handles <think> tags)
        analysis = _extract_content_from_response(raw_content)
        
        # CRITICAL FALLBACK: If extraction returns empty but we have content, use raw
        if not analysis and raw_content:
            logger.warning(f"CRITICAL: Extraction failed for {agent_type}, using raw content")
            logger.debug(f"Raw content first 500 chars: {raw_content[:500]}")
            analysis = raw_content
        
        # Final fallback: return warning if still empty
        if not analysis:
            logger.error(f"{agent_type}: Empty analysis AND empty raw content")
            return (
                f"[WARNING] Unable to generate {agent_type} analysis. The model returned an empty response. This may be due to timeout or insufficient data.",
                {},
                None  # Not an error per se, just empty - return structured warning
            )
        
        # Get usage info
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        
        logger.info(f"{agent_type}: Successfully extracted analysis (length: {len(analysis)})")
        return analysis, usage, None
        
    except TimeoutError as e:
        logger.error(f"{agent_type} timeout: {e}")
        return (
            f"[ERROR] {agent_type} agent timed out after {timeout} seconds. Please try again or increase timeout.",
            {},
            str(e)
        )
    except Exception as e:
        logger.error(f"{agent_type} error: {e}")
        return (
            f"[ERROR] {agent_type} agent failed: {str(e)}",
            {},
            str(e)
        )


def _validate_analysis_response(
    analysis: str,
    agent_type: str,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate and structure an analysis response.
    Ensures a response always has usable content.
    """
    if error:
        return {
            "analysis": analysis if analysis else f"[ERROR] {agent_type} agent failed: {error}",
            "success": False,
            "error": error
        }
    
    if not analysis or not analysis.strip():
        return {
            "analysis": f"[WARNING] Unable to generate {agent_type} analysis. Insufficient data or empty response.",
            "success": False,
            "error": "Empty analysis"
        }
    
    return {
        "analysis": analysis,
        "success": True,
        "error": None
    }


def _make_llm_call_streaming(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    agent_type: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Make streaming LLM call using OpenAI SDK.
    
    Yields:
        Dict with streaming events:
        - {"type": "chunk", "content": "...", "agent": "..."} - Partial content
        - {"type": "reasoning", "content": "...", "agent": "..."} - Reasoning content (DeepSeek R1)
        - {"type": "complete", "analysis": "...", "usage": {...}, "success": True} - Final result
        - {"type": "error", "error": "...", "agent": "..."} - Error occurred
    """
    logger.info(f"Starting {agent_type} streaming LLM call with model {model}")
    
    try:
        client = _get_openai_client(api_key)
        
        # Start streaming request
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/xendex-mcp-server",
                "X-Title": f"Xendex MCP {agent_type.replace('_', ' ').title()}",
            },
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=timeout,
            max_tokens=max_tokens,
            stream=True,
        )
        
        # Accumulate content as we stream
        accumulated_content = ""
        accumulated_reasoning = ""
        chunk_count = 0
        
        for chunk in stream:
            chunk_count += 1
            
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                
                # Check for content
                if delta.content:
                    accumulated_content += delta.content
                    yield {
                        "type": "chunk",
                        "content": delta.content,
                        "agent": agent_type,
                        "chunk_index": chunk_count
                    }
                
                # Check for reasoning (DeepSeek R1 via OpenRouter)
                reasoning_delta = getattr(delta, 'reasoning', None)
                if reasoning_delta:
                    accumulated_reasoning += reasoning_delta
                    yield {
                        "type": "reasoning",
                        "content": reasoning_delta,
                        "agent": agent_type,
                        "chunk_index": chunk_count
                    }
        
        logger.info(f"{agent_type}: Streaming completed. Total chunks: {chunk_count}, Content length: {len(accumulated_content)}")
        
        # Use reasoning as content if content is empty (DeepSeek R1 behavior)
        raw_content = accumulated_content
        if not raw_content and accumulated_reasoning:
            logger.info(f"{agent_type}: Content empty, using accumulated reasoning (length: {len(accumulated_reasoning)})")
            raw_content = accumulated_reasoning
        
        # Extract final analysis
        analysis = _extract_content_from_response(raw_content)
        
        # CRITICAL FALLBACK: If extraction returns empty but we have content, use raw
        if not analysis and raw_content:
            logger.warning(f"CRITICAL: Extraction failed for {agent_type} streaming, using raw content")
            analysis = raw_content
        
        # Final fallback: return warning if still empty
        if not analysis:
            logger.error(f"{agent_type}: Empty analysis AND empty raw content from streaming")
            analysis = f"[WARNING] Unable to generate {agent_type} analysis. The model returned an empty response."
        
        # Yield final complete event
        yield {
            "type": "complete",
            "analysis": analysis,
            "agent": agent_type,
            "success": True,
            "usage": {},  # Usage not available in streaming mode
            "chunk_count": chunk_count,
            "raw_content_length": len(raw_content),
            "reasoning_length": len(accumulated_reasoning)
        }
        
    except TimeoutError as e:
        logger.error(f"{agent_type} streaming timeout: {e}")
        yield {
            "type": "error",
            "error": f"{agent_type} agent timed out after {timeout} seconds",
            "agent": agent_type,
            "success": False
        }
    except Exception as e:
        logger.error(f"{agent_type} streaming error: {e}")
        yield {
            "type": "error",
            "error": f"{agent_type} agent failed: {str(e)}",
            "agent": agent_type,
            "success": False
        }



def _extract_content_from_response(content: str) -> str:
    """Extract actual analysis content from AI response, handling DeepSeek R1 reasoning tags."""
    if not content:
        return ""
    
    # DeepSeek R1 wraps reasoning in <think> tags
    # Case 1: Content is after </think> tag (preferred format)
    if "</think>" in content:
        parts = content.split("</think>")
        if len(parts) > 1:
            actual_content = parts[-1].strip()
            if actual_content:
                return actual_content
        
        # Case 2: If nothing after </think>, extract from inside <think>...</think>
        # This happens when the model puts everything in the thinking block
        if "<think>" in content:
            # Use regex to get content between first <think> and last </think>
            import re
            pattern = r'<think>(.*?)</think>'
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Get the last match (in case of multiple think blocks)
                inner_content = matches[-1].strip()
                if inner_content:
                    print(f"DEBUG: Extracting content from inside <think> tags (length: {len(inner_content)})")
                    return inner_content
    
    # Case 3: No think tags, or extraction failed - return everything
    # This is a fallback to ensure we never lose content
    result = content.strip()
    if result:
        print(f"DEBUG: No think tags found or extraction failed, returning raw content (length: {len(result)})")
    return result


def _get_session_store(session_id: str) -> Dict[str, Any]:
    if session_id not in _SESSION_STATE:
        _SESSION_STATE[session_id] = {}
    return _SESSION_STATE[session_id]


def _ensure_json_serializable(value: Any) -> None:
    # Value comes from MCP arguments (JSON), but validate to avoid surprises.
    json.dumps(value)

