"""
Digital Truth Guardian - Main Entry Point

Provides CLI and API interfaces for the fact-checking system.
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.graph import get_truth_guardian_graph, TruthGuardianGraph
from .core.config import settings
from .database.qdrant_client import get_qdrant_manager
from .utils.logger import setup_logger, get_logger


console = Console()
logger = get_logger()


# ==================== CLI Interface ====================

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(debug: bool):
    """🛡️ Digital Truth Guardian - Combat Misinformation with AI"""
    if debug:
        setup_logger(level="DEBUG")


@cli.command()
@click.argument("claim")
@click.option("--session", "-s", default=None, help="Session ID for conversation")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def verify(claim: str, session: Optional[str], json_output: bool):
    """Verify a claim or statement."""
    asyncio.run(_verify_claim(claim, session, json_output))


async def _verify_claim(claim: str, session: Optional[str], json_output: bool):
    """Async verification handler."""
    console.print(Panel(
        f"[bold blue]Verifying:[/bold blue] {claim}",
        title="🛡️ Digital Truth Guardian",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing claim...", total=None)
        
        try:
            graph = get_truth_guardian_graph()
            result = await graph.verify(claim, session)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            return
    
    if json_output:
        import json
        console.print_json(json.dumps(result, indent=2, default=str))
    else:
        _display_result(result)


def _display_result(result: dict):
    """Display verification result in rich format."""
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0)
    
    # Verdict styling
    verdict_styles = {
        "TRUE": ("green", "✅"),
        "FALSE": ("red", "❌"),
        "UNCERTAIN": ("yellow", "⚠️"),
        "PENDING": ("dim", "⏳")
    }
    
    style, emoji = verdict_styles.get(verdict, ("dim", "❓"))
    
    # Create verdict panel
    verdict_text = f"{emoji} [bold {style}]{verdict}[/bold {style}]"
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    
    console.print()
    console.print(Panel(
        f"[bold]Verdict:[/bold] {verdict_text}\n"
        f"[bold]Confidence:[/bold] {confidence_bar} {int(confidence * 100)}%",
        title="Result",
        border_style=style
    ))
    
    # Display response
    response = result.get("response", "No response generated")
    console.print(Panel(
        Markdown(response),
        title="Analysis",
        border_style="blue"
    ))
    
    # Display sources
    sources = result.get("sources", [])
    if sources:
        table = Table(title="Sources")
        table.add_column("Source", style="cyan")
        for source in sources[:5]:
            table.add_row(source)
        console.print(table)
    
    # Processing info
    console.print(f"\n[dim]Processing time: {result.get('processing_time', 0):.2f}s[/dim]")
    
    if result.get("memory_written"):
        console.print("[green]✓ Saved to knowledge base[/green]")


@cli.command()
def chat():
    """Start interactive chat mode."""
    asyncio.run(_chat_mode())


async def _chat_mode():
    """Interactive chat mode."""
    console.print(Panel(
        "[bold]Welcome to Digital Truth Guardian![/bold]\n\n"
        "Enter claims or questions to verify. Commands:\n"
        "  /quit - Exit chat\n"
        "  /stats - Show knowledge base stats\n"
        "  /clear - Clear session",
        title="🛡️ Interactive Mode",
        border_style="green"
    ))
    
    graph = get_truth_guardian_graph()
    session_id = f"chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        if not query.strip():
            continue
        
        if query.startswith("/"):
            if query == "/quit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif query == "/stats":
                await _show_stats()
                continue
            elif query == "/clear":
                session_id = f"chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                console.print("[green]Session cleared[/green]")
                continue
            else:
                console.print("[yellow]Unknown command[/yellow]")
                continue
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            
            try:
                result = await graph.verify(query, session_id)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        
        console.print(f"\n[bold green]Guardian:[/bold green]")
        console.print(Markdown(result.get("response", "I couldn't process that.")))


async def _show_stats():
    """Display knowledge base statistics."""
    try:
        qdrant = get_qdrant_manager()
        info = await qdrant.get_collection_info()
        stats = await qdrant.get_verdict_stats()
        
        table = Table(title="Knowledge Base Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Records", str(info.get("points_count", 0)))
        table.add_row("Status", info.get("status", "unknown"))
        
        for verdict, count in stats.items():
            table.add_row(f"  {verdict}", str(count))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error fetching stats: {e}[/red]")


@cli.command()
def init():
    """Initialize the database and collections."""
    asyncio.run(_init_database())


async def _init_database():
    """Initialize all Qdrant collections."""
    console.print("[yellow]Initializing database...[/yellow]")
    
    try:
        # Import memory manager
        from .database.memory_manager import get_memory_manager, init_memory_collections
        
        # Initialize knowledge base collection
        qdrant = get_qdrant_manager()
        kb_created = await qdrant.ensure_collection_exists()
        if kb_created:
            console.print("[green]✓ Knowledge base collection created[/green]")
        else:
            console.print("[dim]✓ Knowledge base collection exists[/dim]")
        
        # Initialize memory collections (episodic + shared context)
        memory_results = await init_memory_collections()
        
        for collection_name, was_created in memory_results.items():
            if was_created:
                console.print(f"[green]✓ {collection_name} collection created[/green]")
            else:
                console.print(f"[dim]✓ {collection_name} collection exists[/dim]")
        
        # Show collection info
        console.print()
        info = await qdrant.get_collection_info()
        console.print(f"[cyan]Knowledge Base:[/cyan]")
        console.print(f"  Status: {info.get('status')}")
        console.print(f"  Points: {info.get('points_count', 0)}")
        
        memory_manager = get_memory_manager()
        memory_stats = await memory_manager.get_memory_stats()
        
        console.print(f"\n[cyan]Memory Collections:[/cyan]")
        for coll_name, stats in memory_stats.items():
            console.print(f"  {coll_name}: {stats.get('points_count', 0)} points")
        
        console.print("\n[green]✓ All collections initialized successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    console.print(f"[green]Starting API server on {host}:{port}[/green]")
    uvicorn.run(
        "digital_truth_guardian.api:app",
        host=host,
        port=port,
        reload=reload
    )


# ==================== FastAPI Application ====================

def create_api():
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(
        title="Digital Truth Guardian API",
        description="A Self-Correcting Agentic Immune System for Digital Trust",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request/Response models
    class VerifyRequest(BaseModel):
        claim: str
        session_id: Optional[str] = None
    
    class VerifyResponse(BaseModel):
        query: str
        verdict: str
        confidence: float
        explanation: str
        response: str
        sources: List[str]
        memory_written: bool
        processing_time: float
    
    class HealthResponse(BaseModel):
        status: str
        version: str
        database_status: str
    
    # Endpoints
    @app.get("/", tags=["Info"])
    async def root():
        return {
            "name": "Digital Truth Guardian",
            "version": "1.0.0",
            "description": "A Self-Correcting Agentic Immune System for Digital Trust"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Info"])
    async def health():
        try:
            qdrant = get_qdrant_manager()
            info = await qdrant.get_collection_info()
            db_status = info.get("status", "unknown")
        except:
            db_status = "disconnected"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            database_status=db_status
        )
    
    @app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
    async def verify_claim(request: VerifyRequest):
        """Verify a claim and return the verdict."""
        try:
            graph = get_truth_guardian_graph()
            result = await graph.verify(
                request.claim,
                request.session_id
            )
            
            return VerifyResponse(
                query=result["query"],
                verdict=result["verdict"] or "UNKNOWN",
                confidence=result["confidence"] or 0.0,
                explanation=result["explanation"] or "",
                response=result["response"] or "",
                sources=result["sources"] or [],
                memory_written=result["memory_written"],
                processing_time=result["processing_time"]
            )
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats", tags=["Info"])
    async def get_stats():
        """Get knowledge base statistics."""
        try:
            qdrant = get_qdrant_manager()
            info = await qdrant.get_collection_info()
            stats = await qdrant.get_verdict_stats()
            
            return {
                "collection": info,
                "verdict_counts": stats
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create API app instance
api_app = create_api()


# ==================== Entry Point ====================

def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
