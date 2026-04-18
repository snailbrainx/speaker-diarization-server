"""
MCP (Model Context Protocol) API Endpoint

Standard MCP server over HTTP (SSE transport) at /mcp endpoint.

Allows AI agents to connect from anywhere on the network:
- Flowise: {"url": "http://10.x.x.x:8418/mcp", "transport": "http"}
- Other MCP clients via HTTP

Features:
- Get latest conversation transcripts with speaker labels
- Identify unknown speakers
- Manage speaker profiles
- Query conversation history

Transport: HTTP with Server-Sent Events (SSE)
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, Any, Dict
import inspect
import json
import asyncio
import os
from datetime import datetime

from .database import get_db
from .models import Speaker, Conversation, ConversationSegment


router = APIRouter(prefix="/mcp", tags=["MCP"])

# Store active SSE connections for ping
active_connections: Dict[str, bool] = {}


# ============================================================================
# MCP Tool Implementations
# ============================================================================

async def get_latest_segments(
    conversation_id: Optional[int] = None,
    limit: int = 20,
    db: Session = None
) -> dict:
    """
    Get latest conversation segments with speaker labels.

    Returns segment IDs, speaker names, transcript text, and timestamps.
    Use this to see what was just said and find segments that need identification.
    """
    if conversation_id is None:
        # Get most recent conversation
        conv = db.query(Conversation).order_by(Conversation.id.desc()).first()
        if not conv:
            return {"error": "No conversations found"}
        conversation_id = conv.id

    # Get conversation with segments
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        return {"error": f"Conversation {conversation_id} not found"}

    # Get segments
    segments = db.query(ConversationSegment)\
        .filter(ConversationSegment.conversation_id == conversation_id)\
        .order_by(ConversationSegment.id.desc())\
        .limit(limit)\
        .all()

    # Reverse to chronological order
    segments = list(reversed(segments))

    return {
        "conversation_id": conversation_id,
        "title": conv.title,
        "total_segments": conv.num_segments,
        "returned_segments": len(segments),
        "segments": [
            {
                "id": seg.id,
                "speaker_name": seg.speaker_name or "Unknown",
                "text": seg.text or "",
                "start_offset": seg.start_offset,
                "end_offset": seg.end_offset,
                "confidence": seg.confidence
            }
            for seg in segments
        ]
    }


async def identify_speaker_in_segment(
    conversation_id: int,
    segment_id: int,
    speaker_name: str,
    auto_enroll: bool = True,
    db: Session = None,
) -> dict:
    """
    Identify or correct speaker in a segment.

    This is the KEY tool for speaker identification. When user says
    "Unknown_01 is Bob", call this with the segment ID and "Bob" as speaker_name.

    System will automatically update all matching past segments.
    """
    from .conversation_api import identify_speaker_in_segment as _identify
    from .schemas import IdentifySpeakerRequest
    from .api import get_engine

    try:
        result = await _identify(
            conversation_id=conversation_id,
            segment_id=segment_id,
            request=IdentifySpeakerRequest(speaker_name=speaker_name, enroll=auto_enroll),
            db=db,
            engine=get_engine(),
        )
    except HTTPException as e:
        return {"error": f"Failed to identify speaker: {e.detail}", "status_code": e.status_code}

    updated = result.get("segments_updated", 0) if isinstance(result, dict) else 0
    return {
        "success": True,
        "speaker_name": speaker_name,
        "enrolled": auto_enroll,
        "segments_updated": updated,
        "message": f"Successfully identified speaker as '{speaker_name}'." +
                  (f" Updated {updated} past segments automatically." if updated else ""),
        "details": result,
    }


async def list_speakers(db: Session) -> dict:
    """Get all enrolled speakers"""
    speakers = db.query(Speaker).all()
    return {
        "speakers": [
            {
                "id": s.id,
                "name": s.name,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            }
            for s in speakers
        ]
    }


async def get_conversation(conversation_id: int, db: Session) -> dict:
    """Get full conversation with all segments"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        return {"error": f"Conversation {conversation_id} not found"}

    segments = db.query(ConversationSegment)\
        .filter(ConversationSegment.conversation_id == conversation_id)\
        .order_by(ConversationSegment.id)\
        .all()

    return {
        "id": conv.id,
        "title": conv.title,
        "start_time": conv.start_time.isoformat() if conv.start_time else None,
        "duration": conv.duration,
        "num_speakers": conv.num_speakers,
        "segments": [
            {
                "id": seg.id,
                "speaker_name": seg.speaker_name,
                "text": seg.text,
                "start_offset": seg.start_offset,
                "end_offset": seg.end_offset,
                "confidence": seg.confidence
            }
            for seg in segments
        ]
    }


async def list_conversations(skip: int = 0, limit: int = 10, db: Session = None) -> dict:
    """Get list of conversations"""
    convs = db.query(Conversation)\
        .order_by(Conversation.id.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return {
        "conversations": [
            {
                "id": c.id,
                "title": c.title,
                "start_time": c.start_time.isoformat() if c.start_time else None,
                "duration": c.duration,
                "num_speakers": c.num_speakers,
                "num_segments": c.num_segments
            }
            for c in convs
        ]
    }


async def rename_speaker(speaker_id: int, new_name: str, db: Session = None) -> dict:
    """Rename speaker (updates all past segments)"""
    from .api import rename_speaker as _rename
    from .schemas import SpeakerRename

    try:
        speaker = await _rename(speaker_id=speaker_id, rename_data=SpeakerRename(new_name=new_name), db=db)
    except HTTPException as e:
        return {"error": f"API error: {e.detail}", "status_code": e.status_code}

    return {"id": speaker.id, "name": speaker.name}


async def delete_speaker(speaker_id: int, db: Session = None) -> dict:
    """Delete speaker profile"""
    from .api import delete_speaker as _delete

    try:
        return await _delete(speaker_id=speaker_id, db=db)
    except HTTPException as e:
        return {"error": f"API error: {e.detail}", "status_code": e.status_code}


async def reprocess_conversation(conversation_id: int, db: Session = None) -> dict:
    """Re-analyze conversation with current speaker profiles"""
    from .conversation_api import reprocess_conversation as _reprocess
    from .api import get_engine

    try:
        return await _reprocess(conversation_id=conversation_id, db=db, engine=get_engine())
    except HTTPException as e:
        return {"error": f"API error: {e.detail}", "status_code": e.status_code}


async def update_conversation_title(conversation_id: int, title: str, db: Session = None) -> dict:
    """Update conversation title"""
    from .conversation_api import update_conversation as _update
    from .schemas import ConversationUpdate

    try:
        conv = await _update(
            conversation_id=conversation_id,
            update_data=ConversationUpdate(title=title),
            db=db,
        )
    except HTTPException as e:
        return {"error": f"API error: {e.detail}", "status_code": e.status_code}

    return {"id": conv.id, "title": conv.title}


async def delete_all_unknown_speakers(db: Session) -> dict:
    """
    Delete all unknown speakers (names starting with 'Unknown_').

    Useful for cleanup after identifying all speakers in conversations.
    Returns count of deleted speakers.
    """
    from .services import delete_unknown_speakers

    count, names = delete_unknown_speakers(db)
    db.commit()

    if not count:
        return {"message": "No unknown speakers found", "deleted_count": 0}

    return {
        "success": True,
        "deleted_count": count,
        "deleted_speakers": names,
        "message": f"Successfully deleted {count} unknown speaker(s): {', '.join(names[:5])}" +
                   (f" and {count-5} more" if count > 5 else "")
    }


async def search_conversations_by_speaker(speaker_name: str, db: Session, limit: int = 50, skip: int = 0) -> dict:
    """
    Search for all conversations where a specific speaker appears.

    Useful for querying conversation history by speaker:
    - "What was I talking about last week?"
    - "Show me all conversations with Nick"
    - "When did Andy last speak?"

    Returns conversation IDs, titles, datetimes, durations, and segment counts.
    Ordered by most recent first.

    Args:
        speaker_name: Name of speaker to search for (must exist in database)
        limit: Maximum number of conversations to return (default: 50)
        skip: Number of conversations to skip for pagination (default: 0)

    Returns:
        Dict with speaker info and list of conversations with metadata:
        - conversation_id: ID for use in other tools
        - title: Conversation title
        - datetime: When conversation started (ISO format)
        - duration_minutes: Length of conversation
        - speaker_segments: Number of times this speaker spoke
        - total_segments: Total segments in conversation

    Raises:
        Error if speaker doesn't exist in database
    """
    from .models import Speaker, Conversation, ConversationSegment
    from sqlalchemy import func

    # Validate speaker exists
    speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
    if not speaker:
        return {"error": f"Speaker '{speaker_name}' not found. Use list_speakers tool to see available speakers."}

    # Get all distinct conversations where this speaker appears
    # Join ConversationSegment with Conversation to get conversation details
    conversations_query = (
        db.query(Conversation)
        .join(ConversationSegment, Conversation.id == ConversationSegment.conversation_id)
        .filter(ConversationSegment.speaker_id == speaker.id)
        .distinct()
        .order_by(Conversation.start_time.desc())  # Most recent first
    )

    total_count = conversations_query.count()
    conversations = conversations_query.offset(skip).limit(limit).all()

    # Format results
    conversation_list = []
    for conv in conversations:
        # Count segments by this speaker in this conversation
        speaker_segment_count = (
            db.query(func.count(ConversationSegment.id))
            .filter(
                ConversationSegment.conversation_id == conv.id,
                ConversationSegment.speaker_id == speaker.id
            )
            .scalar()
        )

        conversation_list.append({
            "conversation_id": conv.id,
            "title": conv.title or f"Conversation {conv.id}",
            "datetime": conv.start_time.isoformat() if conv.start_time else None,
            "duration_seconds": conv.duration,
            "duration_minutes": round(conv.duration / 60, 1) if conv.duration else None,
            "total_segments": conv.num_segments,
            "speaker_segments": speaker_segment_count,
            "audio_path": conv.audio_path
        })

    return {
        "speaker_name": speaker.name,
        "speaker_id": speaker.id,
        "total_conversations": total_count,
        "returned_count": len(conversation_list),
        "conversations": conversation_list
    }


# ============================================================================
# Tool Registry
# ============================================================================

TOOLS = {
    "list_conversations": {
        "function": list_conversations,
        "description": "Get list of conversations with their IDs. Returns: conversation_id, title, start_time, num_speakers, num_segments. Use this FIRST to get conversation_id for other tools. Params: skip (offset, default 0), limit (max results, default 10).",
        "params": ["skip", "limit"]
    },
    "get_latest_segments": {
        "function": get_latest_segments,
        "description": "Get recent conversation segments with speaker names, transcript text, and IDs. Returns: segment_id, speaker_name (e.g. 'Unknown_01'), text, timestamps. Use conversation_id from list_conversations or leave empty for most recent. Use this to see what was said and find segment_id for identification. Params: conversation_id (optional, uses latest if not provided), limit (default 20).",
        "params": ["conversation_id", "limit"]
    },
    "identify_speaker_in_segment": {
        "function": identify_speaker_in_segment,
        "description": "Identify/rename speaker in a specific segment. KEY TOOL when user says 'Unknown_01 is Bob'. AUTOMATICALLY updates ALL past segments with same speaker. Creates speaker profile if auto_enroll=true. Returns: success status, segments_updated count. NO NEED to call reprocess_conversation after this - updates are automatic! Params: conversation_id, segment_id (from get_latest_segments), speaker_name, auto_enroll (default true).",
        "params": ["conversation_id", "segment_id", "speaker_name", "auto_enroll"]
    },
    "list_speakers": {
        "function": list_speakers,
        "description": "Get all enrolled speakers with their IDs. Returns: speaker_id, name, created_at, updated_at. Use this to get speaker_id for rename_speaker or delete_speaker tools.",
        "params": []
    },
    "rename_speaker": {
        "function": rename_speaker,
        "description": "Rename an existing speaker profile. AUTOMATICALLY updates ALL past segments with this speaker. Use list_speakers first to get speaker_id. This is useful when speaker already has a profile but wrong name. Returns: updated speaker details. NO NEED to call reprocess_conversation after - updates are automatic! Params: speaker_id (from list_speakers), new_name.",
        "params": ["speaker_id", "new_name"]
    },
    "delete_speaker": {
        "function": delete_speaker,
        "description": "Delete a speaker profile permanently. Use list_speakers first to get speaker_id. Segments referencing this speaker will become unidentified. Returns: success message. Params: speaker_id (from list_speakers).",
        "params": ["speaker_id"]
    },
    "delete_all_unknown_speakers": {
        "function": delete_all_unknown_speakers,
        "description": "Cleanup tool: Delete ALL speakers with names starting with 'Unknown_' in one operation. Useful after identifying all unknowns. Returns: count and list of deleted speakers. NO parameters needed.",
        "params": []
    },
    "get_conversation": {
        "function": get_conversation,
        "description": "Get complete conversation with ALL segments and full transcript. Returns: conversation details + all segments with speaker names and text. Use list_conversations first to get conversation_id. Params: conversation_id (from list_conversations).",
        "params": ["conversation_id"]
    },
    "reprocess_conversation": {
        "function": reprocess_conversation,
        "description": "Re-run speaker recognition on entire conversation. ONLY use this when you've enrolled NEW speakers and want to check if they appear in OLD conversations. NOT needed after identify_speaker_in_segment or rename_speaker - those update automatically! Returns: segments count. Params: conversation_id (from list_conversations).",
        "params": ["conversation_id"]
    },
    "update_conversation_title": {
        "function": update_conversation_title,
        "description": "Change conversation title/name. Returns: updated conversation. Params: conversation_id (from list_conversations), title (new title string).",
        "params": ["conversation_id", "title"]
    },
    "search_conversations_by_speaker": {
        "function": search_conversations_by_speaker,
        "description": "Search conversation history by speaker name. Returns all conversations where the speaker appears with IDs, titles, datetimes, and segment counts. Useful for queries like 'What was I talking about last week?' or 'Show me all conversations with Nick'. Returns error if speaker doesn't exist. Results ordered by most recent first. Params: speaker_name (required, must be valid speaker), limit (default 50), skip (default 0 for pagination).",
        "params": ["speaker_name", "limit", "skip"]
    }
}


# ============================================================================
# HTTP Endpoints
# ============================================================================

@router.get("", include_in_schema=False)
async def mcp_info():
    """
    MCP server information endpoint.

    Returns server capabilities and connection info.
    """
    return {
        "name": "Speaker Diarization MCP Server",
        "version": "1.0.0",
        "protocol": "MCP 2024-11-05",
        "transport": "HTTP (SSE)",
        "description": "AI agent interface for speaker diarization system",
        "endpoints": {
            "sse": "/mcp/sse",
            "rpc": "/mcp"
        },
        "capabilities": {
            "tools": True,
            "resources": False,
            "prompts": False
        },
        "tools_count": len(TOOLS),
        "connection": {
            "example": {
                "url": f"http://localhost:{os.getenv('PORT', '8418')}/mcp",
                "transport": "http"
            }
        }
    }


@router.get("/sse", include_in_schema=False)
async def mcp_sse(request: Request):
    """
    MCP SSE endpoint for server-to-client messages.

    Maintains persistent connection for server notifications and ping/keepalive.
    """
    connection_id = f"conn_{datetime.now().timestamp()}"
    active_connections[connection_id] = True

    async def event_stream():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"

            # Keep connection alive with ping every 30 seconds
            while active_connections.get(connection_id, False):
                await asyncio.sleep(30)

                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Send ping
                yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            # Clean up connection
            active_connections.pop(connection_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("", include_in_schema=False)
async def mcp_rpc(body: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """MCP JSON-RPC endpoint"""
    req_id = body.get("id")

    # Validate
    if body.get("jsonrpc") != "2.0":
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": req_id})

    method = body.get("method")
    params = body.get("params", {})

    # Handle methods
    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "speaker-diarization", "version": "1.0.0"}
            },
            "id": req_id
        })

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": tool["description"],
                        "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in tool["params"]}}
                    }
                    for name, tool in TOOLS.items()
                ]
            },
            "id": req_id
        })

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOLS:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                "id": req_id
            })

        tool = TOOLS[tool_name]
        if "db" in inspect.signature(tool["function"]).parameters:
            arguments["db"] = db

        try:
            result = await tool["function"](**arguments)
            if isinstance(result, dict) and "error" in result:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "Tool execution error", "data": result["error"]},
                    "id": req_id
                })

            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {"content": [{"type": "text", "text": str(result)}]},
                "id": req_id
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error"},
                "id": req_id
            })

    else:
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": req_id
        })
