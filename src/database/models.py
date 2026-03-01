"""
Database Models
Defines the database schema for the driver verification system
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import os
import numpy as np


@dataclass
class Driver:
    """Driver model representing an enrolled driver"""
    driver_id: Optional[int] = None
    name: str = ""
    license_number: Optional[str] = None
    category: str = "B"  # Comma-separated categories e.g. "A,B,C"
    biometric_embedding: Optional[np.ndarray] = None
    enrollment_date: Optional[datetime] = None
    email: Optional[str] = None
    status: str = "active"
    photo_path: Optional[str] = None
    dob: Optional[datetime] = None
    gender: Optional[str] = None
    expiry_date: Optional[datetime] = None
    issue_place: Optional[str] = None

    # Category labels for display
    CATEGORY_LABELS = {
        'B': 'Passenger cars (standard)',
        'C': 'Trucks / heavy goods vehicles',
        'D': 'Buses / passenger transport',
        'E': 'Articulated / special vehicles',
        'F' : 'Tractors and agricultural machinery'
    }

    @property
    def categories(self) -> list:
        """Return sorted list of category codes, e.g. ['A', 'B', 'C']"""
        if not self.category:
            return ['B']
        return sorted(set(c.strip().upper() for c in self.category.split(',') if c.strip()))

    @property
    def categories_display(self) -> str:
        """Return human-readable category string, e.g. 'A, B, C'"""
        return ', '.join(self.categories)

    def to_dict(self) -> dict:
        """Convert driver to dictionary"""
        return {
            'driver_id': self.driver_id,
            'name': self.name,
            'license_number': self.license_number,
            'category': self.category,          # raw DB value: "A,B,C"
            'categories': self.categories,       # parsed list: ["A","B","C"]
            'categories_display': self.categories_display,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None,
            'email': self.email,
            'status': self.status,
            'photo_url': f'/api/driver-photo/{self.driver_id}' if self.photo_path and os.path.isfile(self.photo_path) else None,
            'dob': self.dob.isoformat() if self.dob and hasattr(self.dob, 'isoformat') else self.dob,
            'gender': self.gender,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date and hasattr(self.expiry_date, 'isoformat') else self.expiry_date,
            'issue_place': self.issue_place,
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
    system_id: Optional[str] = None
    brightness: Optional[float] = None
    location: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert log to dictionary"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'driver_id': self.driver_id,
            'driver_name': self.driver_name,
            'similarity_score': float(self.similarity_score or 0),
            'authorized': bool(self.authorized),
            'processing_time_ms': float(self.processing_time_ms or 0),
            'image_path': self.image_path,
            'liveness_passed': bool(self.liveness_passed),
            'system_id': self.system_id,
            'brightness': float(self.brightness) if self.brightness is not None else None,
            'location': self.location,
            'retry_count': int(self.retry_count),
            'unix_ts': self.timestamp.timestamp() if self.timestamp else 0.0
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
            'image_path': self.image_path if self.image_path else '',
            'system_id': self.system_id or '',
            'brightness': f"{self.brightness:.1f}" if self.brightness is not None else '',
            'location': self.location or '',
            'retry_count': self.retry_count
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
