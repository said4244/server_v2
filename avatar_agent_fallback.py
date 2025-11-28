"""
Enhanced Tavus Avatar Agent with proper cleanup and error handling
"""

from dotenv import load_dotenv
#from flask import session
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent
from livekit.plugins import openai, tavus, elevenlabs, silero
from livekit import api
import os
import logging
import asyncio
import json
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection
import signal
import time
import threading
import psutil
import atexit
from AgentInstructions import DebugAvatarAgent
import subprocess

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
EXPECTED_USER_IDENTITY = os.getenv("EXPECTED_USER_IDENTITY")
AVATAR_LANGUAGE = os.getenv("AVATAR_LANGUAGE")
AVATAR_LANGUAGE_STT = os.getenv("AVATAR_LANGUAGE_STT")
replica_id = os.getenv("TAVUS_REPLICA_ID")
persona_id = os.getenv("TAVUS_PERSONA_ID")

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also enable LiveKit debug logging
logging.getLogger("livekit").setLevel(logging.DEBUG)
logging.getLogger("tavus").setLevel(logging.DEBUG)

# Add after the environment variables section:


# Add this function to both avatar_agent.py and avatar_agent_fallback.py

def force_kill_self(room_name: str | None = None, kill_self: bool = False):
    """Kill all avatar agent processes for this room"""
    if room_name:
        try:
            # Get all processes
            result = subprocess.run(
                ["ps", "-eo", "pid,cmd"],
                capture_output=True,
                text=True,
                check=True
            )
            
            pids_to_kill = []
            target_files = ['avatar_agent.py', 'avatar_agent_fallback.py']

            for line in result.stdout.strip().split('\n'):
                if room_name in line and any(f in line for f in target_files):
                    # Extract PID (first field)
                    pid = line.split()[0]
                    try:
                        pid_int = int(pid)
                        # Don't add our own PID
                        if pid_int != os.getpid():
                            pids_to_kill.append(pid_int)
                    except ValueError:
                        continue
            
            # Kill all found PIDs
            for pid in pids_to_kill:
                try:
                    logger.info(f"Killing agent process PID {pid} for room {room_name}")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    logger.info(f"Process {pid} already dead")
                except Exception as e:
                    logger.error(f"Failed to kill PID {pid}: {e}")
                    
            logger.info(f"Killed {len(pids_to_kill)} agent processes for room {room_name}")
            if kill_self:
                os.kill(os.getpid(), signal.SIGKILL)
                subprocess.run(["pkill", "-9", "-f", f"avatar_agent.*{room_name}"], check=False)
                subprocess.run(["pkill", "-9", "-f", f"avatar_agent_fallback.*{room_name}"], check=False)

        except Exception as e:
            logger.error(f"Error in force_kill_self: {e}")
    else: pass

# Signal handler to ensure process dies
def signal_handler(signum, frame):
    logger.error(f"Received signal {signum} - FORCE KILLING")
    os.kill(os.getpid(), signal.SIGKILL)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register atexit handler as final failsafe
def atexit_handler():
    logger.error("ATEXIT: Process still running at exit - FORCE KILLING")
    os.kill(os.getpid(), signal.SIGKILL)

atexit.register(atexit_handler)

async def entrypoint(ctx: agents.JobContext):
    """Main entry point with proper cleanup and error handling"""
    
    logger.info("="*60)
    logger.info(f"Room name: {ctx.room.name}")
    
    avatar = None
    session = None
    
    try:
        # Connect to room
        if hasattr(ctx, 'room_input_options'):
            ctx.room_input_options = agents.RoomInputOptions(close_on_disconnect=False)
        await ctx.connect()
        
        # Create Tavus avatar
        avatar = tavus.AvatarSession(replica_id=replica_id, persona_id=persona_id)

        realtime_model = openai.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview-2024-12-17",
            voice="echo",
            api_key=OPENAI_API_KEY,
            input_audio_transcription=InputAudioTranscription(
                model="gpt-4o-transcribe",
                language=AVATAR_LANGUAGE,
            ),
            temperature=1.0,
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="auto",
                create_response=True,
                interrupt_response=True,
            ),

        )
        
        # Create agent session
        session = AgentSession(llm=realtime_model)

        
        # Start avatar in room
        # This publishes the avatar video to the room
        await avatar.start(session, room=ctx.room)
        
        # Add a small delay to ensure avatar is ready
        await asyncio.sleep(1)
        
        # Start the interactive session
        agent = DebugAvatarAgent(system_type="openai_realtime")
        # Disable automatic close on participant disconnect
        if hasattr(session, '_room_input_options'):
            session._room_input_options = agents.RoomInputOptions(close_on_disconnect=False)
        await session.start(room=ctx.room, agent=agent)

        # --- Listen for data messages ---
        def on_data_received(data_packet: rtc.DataPacket):
            """Handle data messages from Flutter app"""
            async def handle_data():
                try:
                    # Decode the data
                    message = data_packet.data.decode('utf-8')
                    data_obj = json.loads(message)

                    # Check if it's a user message
                    if data_obj.get('type') == 'user_message':
                        content = data_obj.get('content', '')

                        # Process the text prompt through OpenAI Realtime
                        try:
                            # The content is actually instructions for the AI, not a user message
                            # This will make the AI process the prompt and generate appropriate speech
                            await session.generate_reply(instructions=content)
                            
                                
                        except Exception as e:
                            logger.error(f"Error processing prompt: {e}", exc_info=True)
                            
                            # Fallback: Try adding to chat context
                            try:
                                logger.info("Trying fallback method")
                                
                                # Update the agent's instructions temporarily
                                if hasattr(agent, 'instructions'):
                                    original_instructions = agent.instructions
                                    agent.instructions = content
                                    await session.generate_reply()
                                    agent.instructions = original_instructions
                                    logger.info("Fallback method succeeded")
                                else:
                                    logger.error("Could not update agent instructions")
                                    
                            except Exception as fallback_error:
                                logger.error(f"Fallback also failed: {fallback_error}")
                                
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing data message as JSON: {e}")
                    logger.error(f"Raw data: {data_packet.data}")
                except Exception as e:
                    logger.error(f"Error processing data message: {e}", exc_info=True)

            # Schedule the async handler
            asyncio.create_task(handle_data())

        # Register the data received handler
        ctx.room.on("data_received", on_data_received)

        greeting_message = agent.get_greeting_message()
        logger.info("Step 10: Generating initial greeting...")
        try:
            await asyncio.sleep(1.5)
            await session.generate_reply(instructions=greeting_message)

        except Exception as e:
            logger.error(f"Error sending initial greeting: {e}", exc_info=True)


        async def shutdown_now(ctx, session, avatar):
            # Set a hard deadline - kill process in 10 seconds no matter what
            force_kill_self(ctx.room.name)
            def deadline_kill():
                time.sleep(10)
                logger.error("DEADLINE: Shutdown taking too long, FORCE KILLING NOW!")
                force_kill_self(ctx.room.name, kill_self=True)
            
            deadline_thread = threading.Thread(target=deadline_kill, daemon=True)
            deadline_thread.start()
            
            # Try quick cleanup (with very short timeouts)
            try:
                if session:
                    await asyncio.wait_for(
                        session.close() if hasattr(session, 'close') else asyncio.sleep(0),
                        timeout=1.0
                    )
            except:
                pass
            
            try:
                if avatar:
                    if hasattr(avatar, 'close'):
                        await asyncio.wait_for(avatar.close(), timeout=1.0)
                    elif hasattr(avatar, 'disconnect'):
                        await asyncio.wait_for(avatar.disconnect(), timeout=1.0)
            except:
                pass
            
            try:
                if ctx.room and ctx.room.is_connected:
                    await asyncio.wait_for(ctx.room.disconnect(), timeout=1.0)
            except:
                pass
            
            # Don't wait, just kill immediately
            logger.info("Cleanup attempted, FORCE KILLING NOW")
            force_kill_self(ctx.room.name, kill_self=True)

        # Replace the entire monitoring section (after "AGENT FULLY INITIALIZED AND READY!") with:

        # Monitor participants and exit when expected user leaves
        expected_user = EXPECTED_USER_IDENTITY
        logger.info(f"Expected user identity: {expected_user}")
        user_left = asyncio.Event()

        def _on_p_disconnected(p):
            if expected_user and p.identity == expected_user:
                user_left.set()
        ctx.room.on("participant_disconnected", _on_p_disconnected)

        if expected_user and any(p.identity == expected_user for p in ctx.room.remote_participants.values()):
            logger.info(f"Expected user {expected_user} already in room")

        if expected_user:
            logger.info("Waiting for user %s to leave...", expected_user)
            try:
                await user_left.wait()
                logger.info("User %s left; 4s grace then shutdown.", expected_user)
                await asyncio.sleep(2)
                try:
                    await shutdown_now(ctx, session, avatar)
                    force_kill_self(ctx.room.name)
                    subprocess.run(["pkill", "-9", "-f", f"avatar_agent.*{ctx.room.name}"], check=False)
                    subprocess.run(["pkill", "-9", "-f", f"avatar_agent_fallback.*{ctx.room.name}"], check=False)
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}", exc_info=True)
            except asyncio.CancelledError:
                logger.warning("Shutdown cancelled by framework - forcing termination anyway!")
                await shutdown_now(ctx, session, avatar)
                force_kill_self(ctx.room.name)
            return
        else:
            logger.info("No expected user identity set; Failing to continue.")
                
    except Exception as e:
        logger.error(f"ERROR in agent: {type(e).__name__}: {e}")

    finally:
        logger.info("FINALLY BLOCK - ENSURING TERMINATION")
        
        # Start a timer that will kill us in 5 seconds no matter what
        def final_kill():
            time.sleep(5)
            logger.error("FINALLY BLOCK TIMEOUT - FORCE KILLING!")
            force_kill_self(ctx.room.name)
        
        kill_timer = threading.Thread(target=final_kill, daemon=True)
        kill_timer.start()
        
        try:
            # Quick cleanup attempts
            if 'session' in locals() and session:
                try:
                    if hasattr(session, 'close'):
                        await asyncio.wait_for(session.close(), timeout=1)
                except:
                    pass
            if 'avatar' in locals() and avatar:
                try:
                    if hasattr(avatar, 'close'):
                        await asyncio.wait_for(avatar.close(), timeout=1)
                    elif hasattr(avatar, 'disconnect'):
                        await asyncio.wait_for(avatar.disconnect(), timeout=1)
                except:
                    pass
            if ctx.room and hasattr(ctx.room, 'is_connected') and ctx.room.is_connected:
                try:
                    await asyncio.wait_for(ctx.room.disconnect(), timeout=1)
                except:
                    pass
        except:
            pass
        
        logger.info("FINALLY BLOCK COMPLETE - FORCE KILLING NOW")
        
        # Don't wait, just die
        force_kill_self(ctx.room.name, kill_self=True)


if __name__ == "__main__":
    logger.info("Starting Tavus Avatar Agent Worker (Fallback)...")
    
    # Start a failsafe timer that will kill the process after 15 minutes no matter what
    def failsafe_kill():
        time.sleep(900)  # 15 minutes
        logger.error("FAILSAFE: Process still alive after 15 minutes, force killing!")
        os.kill(os.getpid(), signal.SIGKILL)
    
    failsafe_thread = threading.Thread(target=failsafe_kill, daemon=True)
    failsafe_thread.start()
    
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
