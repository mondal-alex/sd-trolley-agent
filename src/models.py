"""Models for the SD Trolley Agent."""

from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Configuration for the trolley agent."""
    name: str = "sd-trolley-agent"
    version: str = "0.1.0"