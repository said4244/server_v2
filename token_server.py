"""
Token server for Tavus Avatar Flutter clients
Generates JWT tokens for connecting to LiveKit rooms
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from datetime import timedelta
import os
from dotenv import load_dotenv
import logging
import json
import subprocess
import psutil
import sys
import shutil
import sqlite3
import threading
import signal
import time
from dataclasses import dataclass
from livekit.protocol import room as room_proto
import secrets



# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    _init_counter_db()
    # (add other startup work here if needed)
    yield
    # --- shutdown ---
    # (add any cleanup if needed; none required for SQLite counter)

app = FastAPI(title="Tavus Avatar Token Server", lifespan=lifespan)

# Configure CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# LiveKit credentials
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://misshuda-pcsvcsuh.livekit.cloud")
MAX_ACTIVE_AGENTS = int(os.getenv("MAX_ACTIVE_AGENTS", 10))  # Limit active agents to prevent abuse
ADMIN_API_KEY = os.getenv("TOKEN_SERVER_API_KEY")
# SQLite counter (replaces counter.txt)
_COUNTER_DB_PATH = os.path.join(os.path.dirname(__file__), "counter.db")
_COUNTER_DB_INIT_RUN = False
_COUNTER_DB_INIT_LOCK = threading.Lock()


# Validate credentials on startup
if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    logger.error("Missing LiveKit credentials in .env file")
    logger.error(f"LIVEKIT_API_KEY: {'✓' if LIVEKIT_API_KEY else '✗'}")
    logger.error(f"LIVEKIT_API_SECRET: {'✓' if LIVEKIT_API_SECRET else '✗'}")
else:
    logger.info("LiveKit credentials loaded successfully")
    logger.info(f"LiveKit URL: {LIVEKIT_URL}")


@dataclass
class AgentProc:
    room: str
    identity: str | None
    popen: subprocess.Popen
    popup: bool  # launched in a visible terminal?
    started_ts: float

# Registry of active agents by room
_AGENT_REGISTRY: dict[str, AgentProc] = {}
_AGENT_LOCK = threading.Lock()

def _register_agent(room: str, identity: str | None, popen: subprocess.Popen, popup: bool):
    with _AGENT_LOCK:
        _AGENT_REGISTRY[room] = AgentProc(room=room, identity=identity, popen=popen, popup=popup, started_ts=time.time())

def _pop_agent(room: str) -> AgentProc | None:
    with _AGENT_LOCK:
        return _AGENT_REGISTRY.pop(room, None)

def _get_agent(room: str) -> AgentProc | None:
    with _AGENT_LOCK:
        return _AGENT_REGISTRY.get(room)
    
def _collect_tree(pid: int) -> list[psutil.Process]:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    procs = parent.children(recursive=True)
    procs.append(parent)
    return procs

def _terminate_tree_posix(pid: int, timeout: float):
    # Send SIGTERM to process group if available
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        # fall back: individual TERM
        for p in _collect_tree(pid):
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
    _wait_for_exit(pid, timeout)
    # Force kill survivors
    for p in _collect_tree(pid):
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

def _terminate_tree_windows(pid: int, timeout: float):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        proc.terminate()
    except psutil.NoSuchProcess:
        return
    _wait_for_exit(pid, timeout)
    # Force kill survivors
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in proc.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass

def _wait_for_exit(pid: int, timeout: float):
    end = time.time() + timeout
    while time.time() < end:
        if not psutil.pid_exists(pid):
            return
        time.sleep(0.1)

def stop_agent(room: str, graceful_timeout: float = 5.0) -> bool:
    """
    Attempt to stop the agent associated with `room`.
    Returns True if process was found (and termination attempted), False otherwise.
    """
    info = _pop_agent(room)
    if not info:
        return False

    pid = info.popen.pid
    logger.info("Stopping agent for room %s (pid=%s)...", room, pid)

    if sys.platform.startswith("win"):
        _terminate_tree_windows(pid, graceful_timeout)
    else:
        _terminate_tree_posix(pid, graceful_timeout)

    return True

def stop_all_agents():
    subprocess.run(["pkill", "-9", "-f", f"avatar_agent.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", f"avatar_agent_fallback.py"], check=False)



def require_admin_key(x_api_key: str = Header(None)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="Server admin key not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden.")

@app.get("/token", dependencies=[Depends(require_admin_key)])
async def create_token(
    identity_id: int = None,
    room_id: int = None,
    language: str = "ar",
    language_stt: str = None
):
    """
    Generate a token for connecting to LiveKit room with Tavus avatar

    Args:
        identity: User identifier (optional)
        room: Room name to join (optional)
    
    Returns:
        JSON with accessToken and connection details
    """
    warnings: list[str] = []

    
    # Validate input parameters - reject invalid characters
    if identity_id is not None and isinstance(identity_id, int) and identity_id > 0:  # Ensure identity_id is an integer
        identity = f"avatar-user-{identity_id}"
        if not room_id: room_id = identity_id
    else:
        if identity_id:
            warnings.append(f"Invalid identity={identity_id!r}; falling back to auto.")
        if room_id is not None and isinstance(room_id, int) and room_id > 0:
            identity_id = room_id
            identity = f"avatar-user-{identity_id}"
        else: 
            identity_id = next_counter_id()
            identity = f"avatar-user-{identity_id}"
            if not room_id:
                room_id = identity_id

    if room_id is not None and isinstance(room_id, int) and room_id > 0:  # Ensure room_id is an integer
        room = f"avatar-room-{room_id}"
    else:
        if room_id:
            warnings.append(f"Invalid room={room_id!r}; falling back to auto.")
        room_id = next_counter_id()
        room = f"avatar-room-{room_id}"
    
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=500, 
            detail="LiveKit credentials not configured. Check .env file."
        )
    
    if len(_AGENT_REGISTRY) >= MAX_ACTIVE_AGENTS and room not in _AGENT_REGISTRY:
        logger.warning("Max active agents reached, cannot start new agent.")
        raise HTTPException(
            status_code=429,
            detail="Max active agents reached"
        )

    try:
        # Create access token
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        
        # Set user identity
        token.with_identity(identity).with_name(f"User-{identity}")
        
        # Grant permissions for avatar interaction
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room,
            can_subscribe=True,      # Subscribe to avatar video/audio
            can_publish=True,        # Publish user audio
            can_publish_data=True,   # For future data channel features
        ))
        
        # Set expiration (24 hours)
        token.with_ttl(timedelta(hours=24))
        
        # Add metadata if needed
        token.with_metadata(
            json.dumps({
                "client": "flutter",
                "avatar_enabled": True
            })
        )
        
        # Generate JWT
        jwt_token = token.to_jwt()
        
        logger.info(f"Token generated for {identity} in room {room}")
        
        # Start new agent for this room
        await start_new_agent(room, identity, language, language_stt)

        response = {
            "accessToken": jwt_token,
            "url": LIVEKIT_URL,
            "room": room,
            "identity": identity,
            "expiresIn": 86400  # 24 hours in seconds
        }
        
        # Include warnings if any
        if warnings:
            response["warnings"] = warnings
            
        return response
        
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Token generation failed: {str(e)}"
        )

@app.get("/health", dependencies=[Depends(require_admin_key)])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "livekit_configured": bool(LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        "livekit_url": LIVEKIT_URL,
        "version": "1.0.0"
    }


@app.get("/", dependencies=[Depends(require_admin_key)])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Tavus Avatar Token Server",
        "description": "Generates tokens for Flutter clients to connect to Tavus avatar sessions",
        "endpoints": {
            "/token": "Get access token for LiveKit room",
            "/health": "Health check",
            "/rooms": "List active rooms (if enabled)",
        },
        "usage": "GET /token?identity_id=int&room_id=int"
    }

@app.get("/_debug/agents", dependencies=[Depends(require_admin_key)])
async def _debug_agents():
    with _AGENT_LOCK:
        data = {r: {"pid": ap.popen.pid, "identity": ap.identity, "popup": ap.popup, "started": ap.started_ts}
                for r, ap in _AGENT_REGISTRY.items()}
    return data

@app.post("/_debug/stop/{room}", dependencies=[Depends(require_admin_key)])
async def _debug_stop(room: str, all: bool = False):
    if not all:
        stopped = stop_agent(room)
        return {"room": room, "stopped": stopped}
    elif all:
        stop_all_agents()
        return {"message": "All agents stopped"}


@app.get("/_debug/logs/server", dependencies=[Depends(require_admin_key)])
async def _debug_logs(lines: int = 100):
    """Get the last N lines from server.log"""
    log_path = os.path.join(os.path.dirname(__file__), "server.log")
    
    try:
        if not os.path.exists(log_path):
            return {"logs": [], "message": "Log file does not exist yet"}
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            
        # Get the last N lines
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.rstrip('\n\r') for line in last_lines],
            "total_lines": len(all_lines),
            "showing_lines": len(last_lines),
            "log_file": log_path
        }
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read log file: {str(e)}"
        )


@app.get("/_debug/logs/fallback", dependencies=[Depends(require_admin_key)])
async def _debug_logs(lines: int = 100):
    """Get the last N lines from fallback.log"""
    log_path = os.path.join(os.path.dirname(__file__), "fallback.log")

    try:
        if not os.path.exists(log_path):
            return {"logs": [], "message": "Log file does not exist yet"}
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            
        # Get the last N lines
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.rstrip('\n\r') for line in last_lines],
            "total_lines": len(all_lines),
            "showing_lines": len(last_lines),
            "log_file": log_path
        }
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read log file: {str(e)}"
        )

@app.get("/rooms", dependencies=[Depends(require_admin_key)])
async def list_rooms():
    """List active rooms (optional endpoint)"""
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured"
        )
    
    lk_api = None
    try:
        # Create LiveKit API client
        lk_api = api.LiveKitAPI(
            LIVEKIT_URL,
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET
        )
        
        # List rooms with proper request object
        list_request = room_proto.ListRoomsRequest()
        rooms_response = await lk_api.room.list_rooms(list_request)
        
        return {
            "rooms": [
                {
                    "name": room.name,
                    "sid": room.sid,
                    "num_participants": room.num_participants,
                    "creation_time": room.creation_time,
                }
                for room in rooms_response.rooms
            ],
            "total": len(rooms_response.rooms)
        }
    except Exception as e:
        logger.error(f"Error listing rooms: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list rooms: {str(e)}"
        )
    finally:
        # Properly close the client session to avoid resource leaks
        if lk_api:
            await lk_api.aclose()

def _init_counter_db():
    """Create counter table if missing."""
    global _COUNTER_DB_INIT_RUN
    if _COUNTER_DB_INIT_RUN:
        return
    with _COUNTER_DB_INIT_LOCK:
        if _COUNTER_DB_INIT_RUN:
            return
        conn = sqlite3.connect(_COUNTER_DB_PATH)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS id_counter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        finally:
            conn.close()
        _COUNTER_DB_INIT_RUN = True

def next_counter_id() -> int:
    """Return next unique integer ID (1-based) using SQLite AUTOINCREMENT."""
    _init_counter_db()
    conn = sqlite3.connect(_COUNTER_DB_PATH, timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO id_counter DEFAULT VALUES;")
        cur = conn.execute("SELECT last_insert_rowid();")
        (val,) = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    return int(val)

async def start_new_agent(room_name: str, identity: str, language: str, language_stt: str):
    """Start a new Avatar agent for a specific room in a new terminal window"""
    
    # Kill existing agent if any for this specific room
    stop_agent(room_name)
    agent_script = os.path.join(os.path.dirname(__file__), "avatar_agent.py")
    python_exe = sys.executable  # assumes we're already in the desired environment
    env = {**os.environ, "EXPECTED_USER_IDENTITY": identity, "AVATAR_LANGUAGE": language, "AVATAR_LANGUAGE_STT": language_stt or language}

    # Local Windows-specific command to start new agent only for testing
    if sys.platform.startswith("win"):
        # Start new agent with room name in new terminal
        cmd = [python_exe, "-u", agent_script, "connect", "--room", room_name]
    
        proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env = env
        )
        _register_agent(room_name, identity, proc, popup=True)
        
    
        logger.info("Started new Avatar agent for room %s in Windows PowerShell terminal (local testing).", room_name)
        return

    # Server Linux-specific command to start new agent in terminal
    SHOW_LINUX_AGENT_TERMINAL = False  #
    log_path = os.path.join(os.path.dirname(__file__), "server.log")
    agent_script = os.path.join(os.path.dirname(__file__), "avatar_agent.py")
    python_exe = sys.executable  # assumes we're already in the desired environment
    agent_cmd = [python_exe, "-u", agent_script, "connect", "--room", room_name]
    log_file = open(log_path, "a", encoding='utf-8')

    popup = False
    if SHOW_LINUX_AGENT_TERMINAL:
        # Try to locate a terminal emulator. Honor $TERMINAL if set, else probe common ones.
        term = os.getenv("TERMINAL")
        if not term:
            for candidate in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
                if shutil.which(candidate):
                    term = candidate
                    break
        if term in ("gnome-terminal", "konsole"):
            full_cmd = [term, "--", *agent_cmd]
            popup = True
            preexec = None
        elif term:
            full_cmd = [term, "-e", *agent_cmd]
            popup = True
            preexec = None
        else:
            # No terminal, headless mode
            full_cmd = agent_cmd
            preexec = os.setsid
        proc = subprocess.Popen(full_cmd, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=preexec, env=env)
    else:
        # Headless/background: just run the agent, logging to server.log.
        proc = subprocess.Popen(agent_cmd, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)

    log_file.close()

    _register_agent(room_name, identity, proc, popup=popup)
    logger.info("Started new Avatar agent for room %s (Linux/POSIX). Output -> %s", room_name, log_path)

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting Tavus Avatar token server on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )