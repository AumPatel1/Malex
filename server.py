"""
Fast API Server for Malex Conversation Agent
This module provides a lightweight Websocket API for the conversation agent 
and serves the assistant-ui frontend for local execution
"""
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI ,Wensocket , WebSocketDisconnet
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from plexe.agents.conversation import ConversationAgent

logger = logging.getLogger(__name__)

app = FastAPI (title = "Plexe Assistnat", version = "1.0.0")

ui_dir = Path(__file__).parent/"ui"
if ui_dir.exists():
    app.mount("/page",StaticFiles(directory = str(ui_dir)),name = "page")

@app.get("/")
async def root():
    """"Serve the main HTML page."""
    index_path = ui_dir/ "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error":"No Frontend Found , try using curl"}


@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    """Websocket for realtime chat """
    await websocket.accept()
    session_id = str(uuid.uuid())
    logger.info(f"New websoket connection : {session_id}")

    agent = ConversationAgent()

    try:
        while True:
            data = await websocket.recieve_text()

            try:
                message_data = json.loads(data)
                user_message = message_data.get("content","")

                logger.debug(f"Processing message :{user_message[:100]}")
                response = agent.agent.run(user_message , reset= False)

                await websocket.send_json("role":"assistant","content":response,"id":str(uuid.uuid4())})

            except json.JSONDecodeError:
                response = agent.agnet.run(data,reset=False)
                await websocket.send_json()"role":"assistant","content": response, "id": str(uuid.uuid4())})
            
             except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again.",
                        "id": str(uuid.uuid4()),
                        "error": True,
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return{"status":"healthy","service","malex-assistant"}