"""
Centralized Logging Configuration Module

This module implements a centralized logging configuration system for the application.
Key benefits of this approach:

1. Consistent logging format across all modules
2. Centralized control of log levels
3. File rotation to manage log size
4. Both console and file logging
5. Environment-aware configuration

Usage:
    # In module files:
    from src.agent.logging_config import get_logger
    
    # Get a module-specific logger
    logger = get_logger(__name__)
    
    # Use the logger
    logger.info("This is an info message")
    logger.error("This is an error message")
    
    # For main application or standalone scripts:
    from src.agent.logging_config import configure_logging
    
    # Configure logging with defaults
    configure_logging()
    
    # Or with custom settings
    configure_logging(
        level=logging.DEBUG,
        log_dir="custom_logs",
        log_file="myapp.log"
    )

Log Levels:
    - DEBUG: Detailed information, typically for debugging
    - INFO: Confirmation that things are working as expected
    - WARNING: An indication that something unexpected happened
    - ERROR: Due to a more serious problem, the software has not been able to perform a function
    - CRITICAL: A serious error, indicating that the program itself may be unable to continue running
"""
import logging
import logging.handlers
import os
from typing import Optional

def configure_logging(
    level: int = logging.INFO, 
    log_dir: str = "logs", 
    log_file: Optional[str] = "application.log",
    module_name: Optional[str] = None
):
    """
    Configure logging for the entire application or specific module.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Directory to store log files (default: "logs")
        log_file: Log file name (default: "application.log")
        module_name: If provided, configures logging for this module only
                    otherwise configures the root logger
    
    Returns:
        Logger object (either root logger or module-specific logger)
    """
    # Create log directory if it doesn't exist
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get the appropriate logger
    if module_name:
        logger = logging.getLogger(module_name)
    else:
        logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicates when reconfiguring
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    
    # File handler with rotation (if log_file is specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    
    return logger

def get_logger(module_name: str):
    """
    Get a logger for a specific module. This doesn't configure handlers,
    it just returns the logger object, ensuring we use the same logger
    configuration across the application.
    
    Args:
        module_name: The module name to get a logger for
        
    Returns:
        Logger object for the specified module
    """
    return logging.getLogger(module_name) 