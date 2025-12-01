

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime

from agents.orchestrator import AgentOrchestrator

app = FastAPI(
    title="Vanilar Agent",
    description="A agent system for data analysis, visualization, and presentation generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

orchestrator = AgentOrchestrator(api_key=GEMINI_API_KEY)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    response: Dict[str, Any]
    timestamp: str


class AgentInfo(BaseModel):
    name: str
    capabilities: List[str]


@app.get("/")
async def root():
    return {
        "name": "Multi-Agent Code Interpreter API",
        "version": "1.0.0",
        "status": "running",
        "agents": list(orchestrator.list_agents().keys()),
    }


@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    agents_info = []
    for name, capabilities in orchestrator.list_agents().items():
        agents_info.append(AgentInfo(name=name, capabilities=capabilities))
    return agents_info


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - process natural language queries
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "context": {},
                "history": [],
            }

        session = sessions[session_id]
        
        files = session.get("uploaded_files", None)
        results = await orchestrator.chat(
            message=request.message, files=files, conversation_context=session["context"]
        )

        if results["success"] and results.get("agent_results"):
            for agent_name, agent_result in results["agent_results"].items():
                session["context"][f"{agent_name.lower()}_data"] = agent_result["data"]

        session["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": request.message,
                "results": results,
            }
        )

        return ChatResponse(
            success=results["success"],
            session_id=session_id,
            response=results,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    message: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Get or create session
        session_id = session_id or str(uuid.uuid4())

        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "context": {},
                "history": [],
                "uploaded_files": {},
            }

        session = sessions[session_id]

        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        session["uploaded_files"][file.filename] = str(file_path)

        if message:
            results = await orchestrator.chat(
                message=message,
                files={file.filename: str(file_path)},
                conversation_context=session["context"],
            )

            if results["success"] and results.get("agent_results"):
                for agent_name, agent_result in results["agent_results"].items():
                    session["context"][f"{agent_name.lower()}_data"] = agent_result[
                        "data"
                    ]

            session["history"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "file": file.filename,
                    "results": results,
                }
            )

            return {
                "success": True,
                "session_id": session_id,
                "file_uploaded": file.filename,
                "file_path": str(file_path),
                "response": results,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "success": True,
                "session_id": session_id,
                "file_uploaded": file.filename,
                "file_path": str(file_path),
                "message": "File uploaded successfully. Send a message to analyze it.",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), query: str = Form(...)):
    
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = await orchestrator.process_query(
            query=query, files={file.filename: str(file_path)}
        )

        return {
            "success": results["success"],
            "query": query,
            "file": file.filename,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "session": sessions[session_id]}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if "uploaded_files" in session:
        for filename, filepath in session["uploaded_files"].items():
            try:
                Path(filepath).unlink()
            except:
                pass

    del sessions[session_id]

    return {"success": True, "message": f"Session {session_id} deleted"}


@app.post("/clear")
async def clear_all():
    for session in sessions.values():
        if "uploaded_files" in session:
            for filename, filepath in session["uploaded_files"].items():
                try:
                    Path(filepath).unlink()
                except:
                    pass

    sessions.clear()
    orchestrator.clear_context()

    return {"success": True, "message": "All sessions and context cleared"}


@app.get("/history")
async def get_history():
    return {"history": orchestrator.get_execution_history()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
