# class to log data
class Logger:
    """
    A class for logging data.
    """
    def __init__(self, log_file=None, console_output=True):
        self.log_file = log_file
        self.console_output = console_output
        
        # Initialize log file if provided
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("=== Log Started ===\n")
    
    def log(self, message, level="INFO"):
        """
        Log a message with specified level.
        
        Args:
            message (str): The message to log
            level (str): Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = self._get_timestamp()
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # Output to console if enabled
        if self.console_output:
            print(formatted_message)
        
        # Write to log file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")
    
    def info(self, message):
        """Log an info message"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """Log a warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log an error message"""
        self.log(message, "ERROR")
    
    def debug(self, message):
        """Log a debug message"""
        self.log(message, "DEBUG")
    
    def _get_timestamp(self):
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
