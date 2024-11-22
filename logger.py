import logging
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create timestamp for unique log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log(self, message, level='info'):
        """
        Levels:
        - info: Standard information
        - debug: Detailed debugging information
        - warning: Warning messages
        - error: Error messages
        - metric: Training/validation metrics
        """
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'metric':
            self.logger.info(f"[METRIC] {message}")
        else:
            self.logger.info(message)