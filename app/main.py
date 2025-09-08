"""
FastAPI main application
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app.core.config import get_settings
from app.api.routes import router as api_router


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print(f"🎵 Starting {settings.app_name} v{settings.app_version}")
    
    # Create directories if they don't exist
    os.makedirs(settings.outputs_dir, exist_ok=True)
    os.makedirs(settings.models_dir, exist_ok=True)
    os.makedirs(settings.static_dir, exist_ok=True)
    
    print("✅ Application startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI Music Studio for creating music and vocals using open-source models",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists(settings.static_dir):
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenMusic AI Studio</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { color: #fff; text-align: center; margin-bottom: 30px; }
            .feature {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }
            .api-link {
                display: inline-block;
                background: #4CAF50;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 10px 10px 0;
            }
            .api-link:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 OpenMusic AI Studio</h1>
            <p>Welcome to the AI-powered music creation platform!</p>
            
            <div class="feature">
                <h3>🎼 Music Generation</h3>
                <p>Create instrumental music using advanced AI models</p>
            </div>
            
            <div class="feature">
                <h3>🎤 Vocal Synthesis</h3>
                <p>Generate vocals and speech with state-of-the-art TTS</p>
            </div>
            
            <div class="feature">
                <h3>✍️ Lyric Generation</h3>
                <p>AI-powered lyric writing using large language models</p>
            </div>
            
            <div class="feature">
                <h3>🎚️ Audio Processing</h3>
                <p>Mix, combine, and process your generated audio</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/docs" class="api-link">📚 API Documentation</a>
                <a href="/api/health" class="api-link">🔍 Health Check</a>
                <a href="/api/models/status" class="api-link">🤖 Model Status</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version
    }


def main():
    """Run the application"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    main()