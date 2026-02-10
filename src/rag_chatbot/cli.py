"""Modern async CLI interface for the chatbot."""

import asyncio
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import setup_logging
from rag_chatbot.core.exceptions import (
    ConfigurationError,
    SecurityError,
    InputValidationError,
    PromptInjectionError,
    RateLimitExceededError,
    ServiceError,
)
from rag_chatbot.services.chatbot import get_chatbot_service

console = Console()


def print_error(message: str) -> None:
    """Print error message with styling."""
    console.print(Panel(message, title="Error", border_style="red"))


def print_success(message: str) -> None:
    """Print success message with styling."""
    console.print(Panel(message, title="Success", border_style="green"))


def print_warning(message: str) -> None:
    """Print warning message with styling."""
    console.print(Panel(message, title="Warning", border_style="yellow"))


async def ask_question(question: str) -> None:
    """Ask a question and display the answer.
    
    Args:
        question: User's question
    """
    service = await get_chatbot_service()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching and generating answer...", total=None)
        
        try:
            result = await service.process_question(question)
            progress.update(task, completed=True)
            
            if result["success"]:
                # Display answer with markdown rendering
                console.print("\n")
                console.print(Markdown(result["answer"]))
                console.print("\n")
                
                # Display sources
                if result.get("sources"):
                    sources_text = "\n".join(
                        f"- Document {i+1} (score: {s['score']:.2f})"
                        for i, s in enumerate(result["sources"])
                    )
                    console.print(Panel(sources_text, title="Sources", border_style="blue"))
                
                # Display correlation ID for debugging
                console.print(
                    f"[dim]Correlation ID: {result['correlation_id']}[/dim]"
                )
            else:
                print_error("Failed to process question")
                
        except InputValidationError as e:
            progress.update(task, completed=True)
            print_error(f"Input Error: {e}")
        except PromptInjectionError:
            progress.update(task, completed=True)
            print_error("Security Error: Potentially harmful input detected.")
        except RateLimitExceededError as e:
            progress.update(task, completed=True)
            print_error(f"Rate Limit: {e}\nPlease wait {e.retry_after} seconds and try again.")
        except ServiceError as e:
            progress.update(task, completed=True)
            print_error(f"Service Error: {e}")
        except ConfigurationError as e:
            progress.update(task, completed=True)
            print_error(f"Configuration Error: {e}")


async def check_health() -> None:
    """Check service health status."""
    service = await get_chatbot_service()
    
    with console.status("[bold green]Checking service health..."):
        health = await service.health_check()
    
    status_color = "green" if health["status"] == "healthy" else "yellow"
    console.print(Panel(
        f"Overall Status: [{status_color}]{health['status']}[/{status_color}]\n\n"
        f"Elasticsearch: {health['services']['elasticsearch']['status']}\n"
        f"LLM Service: {health['services']['llm']['status']}",
        title="Health Check",
        border_style=status_color,
    ))


@click.group()
@click.version_option(version="1.0.0")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(), help="Path to config file")
def cli(debug: bool, config: Optional[str]) -> None:
    """RAG Chatbot CLI - Ask questions using AI-powered search."""
    # Setup logging
    level = "DEBUG" if debug else "INFO"
    setup_logging(level=level)


@cli.command()
@click.argument("question")
def ask(question: str) -> None:
    """Ask a question and get an AI-powered answer."""
    try:
        asyncio.run(ask_question(question))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command()
def interactive() -> None:
    """Start interactive chat session."""
    console.print(Panel(
        "Welcome to RAG Chatbot!\n"
        "Type your questions and press Enter.\n"
        "Type 'exit', 'quit', or press Ctrl+C to exit.",
        title="Interactive Mode",
        border_style="green",
    ))
    
    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("exit", "quit", "q"):
                console.print("[green]Goodbye![/green]")
                break
            
            if question.lower() in ("health", "status"):
                asyncio.run(check_health())
                continue
            
            asyncio.run(ask_question(question))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            print_error(f"Error: {e}")


@cli.command()
def health() -> None:
    """Check service health status."""
    try:
        asyncio.run(check_health())
    except Exception as e:
        print_error(f"Health check failed: {e}")
        sys.exit(1)


@cli.command()
def config() -> None:
    """Display current configuration."""
    settings = get_settings()
    
    config_text = f"""
Application:
  Name: {settings.app_name}
  Version: {settings.app_version}
  Environment: {settings.environment}
  Debug: {settings.debug}

Elasticsearch:
  URL: {settings.es_url}
  Index: {settings.es_index}
  Verify SSL: {settings.es_verify_ssl}
  Timeout: {settings.es_timeout}s
  Results: {settings.es_results_size}

OpenAI:
  Model: {settings.openai_model}
  Timeout: {settings.openai_timeout}s
  Temperature: {settings.openai_temperature}
  Max Tokens: {settings.openai_max_tokens}

Security:
  Max Query Length: {settings.max_query_length}
  Rate Limit: {settings.rate_limit_requests}/{settings.rate_limit_window}s

Cache:
  Enabled: {settings.cache_enabled}
  Backend: {settings.cache_backend}
  TTL: {settings.cache_ttl}s
    """
    
    console.print(Panel(config_text, title="Configuration", border_style="cyan"))


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
