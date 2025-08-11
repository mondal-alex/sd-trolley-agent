"""
Main module for the SD Trolley Agent.
"""

from models import AgentConfig

def main():
    """Main entry point for the application."""
    config = AgentConfig()
    print(f"Hello from {config.name} v{config.version}!")
    print("UV setup is complete and ready for development.")


if __name__ == "__main__":
    main()
