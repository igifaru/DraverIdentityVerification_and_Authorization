"""
Database Models
Defines the database schema for the driver verification system
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class Driver:
    """Driver model representing an enrolled driver"""
    driver_id: Optional[int] = None
    name: str = ""
    license_number: Optional[str] = None
    category: str = "A"  # Driver category: A, B, C, D, etc.
    biometric_embedding: Optional[np.ndarray] = None
    enrollment_date: Optional[datetime] = None
    email: Optional[str] = None
    status: str = "active"
    
    def to_dict(self) -> dict:
        """Convert driver to dictionary"""
        return {
            'driver_id': self.driver_id,
            'name': self.name,
            'license_number': self.license_number,
            'category': self.category,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None,
            'email': self.email,
            'status': self.status
        }


@dataclass
class VerificationLog:
    """Verification log model representing a verification attempt"""
    log_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    driver_id: Optional[int] = None
    driver_name: Optional[str] = None
    similarity_score: float = 0.0
    authorized: bool = False
    processing_time_ms: float = 0.0
    image_path: Optional[str] = None
    liveness_passed: bool = False
    
    def to_dict(self) -> dict:
        """Convert log to dictionary"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'driver_id': self.driver_id,
            'driver_name': self.driver_name,
            'similarity_score': self.similarity_score,
            'authorized': self.authorized,
            'processing_time_ms': self.processing_time_ms,
            'image_path': self.image_path,
            'liveness_passed': self.liveness_passed
        }
    
    def to_csv_row(self) -> dict:
        """Convert log to CSV row format"""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else '',
            'driver_id': self.driver_id if self.driver_id else 'UNKNOWN',
            'driver_name': self.driver_name if self.driver_name else 'UNAUTHORIZED',
            'similarity_score': f"{self.similarity_score:.4f}",
            'authorized': 'YES' if self.authorized else 'NO',
            'liveness_passed': 'YES' if self.liveness_passed else 'NO',
            'processing_time_ms': f"{self.processing_time_ms:.2f}",
            'image_path': self.image_path if self.image_path else ''
        }
@dataclass
class SystemAuditLog:
    """System audit log for administrative actions"""
    audit_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    action: str = ""  # e.g., "LOGIN", "START_ENGINE", "STOP_ENGINE"
    user_email: str = ""
    details: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert audit log to dictionary"""
        return {
            'audit_id': self.audit_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'action': self.action,
            'user_email': self.user_email,
            'details': self.details,
            'ip_address': self.ip_address
        }
