"""
Configuration Management Module
Loads and manages system configuration from YAML and environment variables
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any


class Config:
    """Singleton configuration manager"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML and environment variables"""
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Load YAML configuration
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Load environment variables
        env_path = project_root / "config" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Convert relative paths to absolute paths
        self._resolve_paths(project_root)
    
    def _resolve_paths(self, project_root: Path):
        """Convert relative paths in config to absolute paths"""
        if 'database' in self._config and 'path' in self._config['database']:
            db_path = self._config['database']['path']
            if not os.path.isabs(db_path):
                self._config['database']['path'] = str(project_root / db_path)
        
        if 'logging' in self._config:
            if 'log_path' in self._config['logging']:
                log_path = self._config['logging']['log_path']
                if not os.path.isabs(log_path):
                    self._config['logging']['log_path'] = str(project_root / log_path)
            
            if 'alert_image_path' in self._config['logging']:
                alert_path = self._config['logging']['alert_image_path']
                if not os.path.isabs(alert_path):
                    self._config['logging']['alert_image_path'] = str(project_root / alert_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('verification.similarity_threshold')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable"""
        return os.getenv(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy()
    
    @property
    def verification_threshold(self) -> float:
        """Get similarity threshold for verification"""
        return self.get('verification.similarity_threshold', 0.6)
    
    @property
    def liveness_ear_threshold(self) -> float:
        """Get EAR threshold for liveness detection"""
        return self.get('verification.liveness_ear_threshold', 0.25)
    
    @property
    def liveness_blink_frames(self) -> int:
        """Get minimum blink frames for liveness detection"""
        return self.get('verification.liveness_blink_frames', 3)
    
    @property
    def camera_resolution(self) -> tuple:
        """Get camera resolution (width, height)"""
        width = self.get('camera.resolution_width', 640)
        height = self.get('camera.resolution_height', 480)
        return (width, height)
    
    @property
    def camera_fps(self) -> int:
        """Get camera FPS"""
        return self.get('camera.fps', 30)
    
    @property
    def camera_device_id(self) -> int:
        """Get camera device ID"""
        return self.get('camera.device_id', 0)
    
    @property
    def database_path(self) -> str:
        """Get database file path"""
        return self.get('database.path')
    
    @property
    def log_path(self) -> str:
        """Get log file path"""
        return self.get('logging.log_path')
    
    @property
    def alert_image_path(self) -> str:
        """Get alert image directory path"""
        return self.get('logging.alert_image_path')
    
    @property
    def smtp_server(self) -> str:
        """Get SMTP server"""
        return self.get('email.smtp_server', 'smtp.gmail.com')
    
    @property
    def smtp_port(self) -> int:
        """Get SMTP port"""
        return self.get('email.smtp_port', 587)
    
    @property
    def smtp_email(self) -> str:
        """Get SMTP email from environment"""
        return self.get_env('SMTP_EMAIL')
    
    @property
    def smtp_password(self) -> str:
        """Get SMTP password from environment"""
        return self.get_env('SMTP_PASSWORD')
    
    @property
    def alert_recipients(self) -> list:
        """Get alert recipient emails"""
        recipients = self.get_env('ALERT_RECIPIENTS', '')
        if recipients:
            return [email.strip() for email in recipients.split(',')]
        return []
    
    @property
    def debug_mode(self) -> bool:
        """Get debug mode flag"""
        return self.get('system.debug_mode', False)
    
    @property
    def system_id(self) -> str:
        """Get system identifier"""
        return self.get('system.system_id', 'UNKNOWN-ID')
    
    @property
    def vehicle_plate(self) -> str:
        """Get vehicle plate number"""
        return self.get('system.vehicle_plate', 'UNKNOWN-PLATE')
    
    @property
    def owner_name(self) -> str:
        """Get owner name"""
        return self.get('system.owner_name', 'UNKNOWN-OWNER')
    
    def validate(self) -> bool:
        """Validate critical configuration parameters"""
        errors = []
        
        # Check database path
        if not self.database_path:
            errors.append("Database path not configured")
        
        # Check camera settings
        if self.camera_device_id < 0:
            errors.append("Invalid camera device ID")
        
        # Check thresholds
        if not (0 <= self.verification_threshold <= 1):
            errors.append("Verification threshold must be between 0 and 1")
        
        if not (0 <= self.liveness_ear_threshold <= 1):
            errors.append("Liveness EAR threshold must be between 0 and 1")
        
        # Check email configuration (warning only)
        if not self.smtp_email or not self.smtp_password:
            print("WARNING: Email credentials not configured. Alerts will not be sent.")
        
        if errors:
            for error in errors:
                print(f"CONFIG ERROR: {error}")
            return False
        
        return True


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    print(f"Verification threshold: {config.verification_threshold}")
    print(f"Camera resolution: {config.camera_resolution}")
    print(f"Database path: {config.database_path}")
    print(f"Debug mode: {config.debug_mode}")
    print(f"\nValidation: {'PASSED' if config.validate() else 'FAILED'}")
