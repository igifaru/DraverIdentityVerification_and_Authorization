"""
Email Service Module
Handles SMTP email alerts for unauthorized access attempts
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from utils.config import config

class EmailService:
    """Manages email alerts for security events"""
    
    def __init__(self):
        """Initialize email service"""
        self.smtp_server = config.smtp_server
        self.smtp_port = config.smtp_port
        self.sender_email = config.smtp_email
        self.sender_password = config.smtp_password
        self.sender_name = config.get('email.sender_name', 'Driver Verification System')
        
        self.template_path = Path(__file__).parent / "templates" / "email_alert.html"
        
        self.is_configured = self._check_configuration()
        
        if self.is_configured:
            print("[OK] Email service configured")
        else:
            print("WARNING: Email service not configured (alerts will not be sent)")
    
    def _check_configuration(self) -> bool:
        """Check if email is properly configured"""
        if not self.sender_email or not self.sender_password:
            return False
        
        if self.sender_email == "your-email@gmail.com":
            return False
        
        return True
    
    def send_unauthorized_alert(self, 
                               recipients: List[str],
                               similarity_score: float,
                               best_match_name: str = None,
                               image_path: str = None,
                               timestamp: datetime = None) -> bool:
        """
        Send unauthorized access alert email
        
        Args:
            recipients: List of recipient email addresses
            similarity_score: Similarity score from verification
            best_match_name: Name of best matching driver (if any)
            image_path: Path to captured image
            timestamp: Timestamp of the event
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.is_configured:
            print("WARNING: Email not configured, skipping alert")
            return False
        
        if not recipients:
            recipients = config.alert_recipients
        
        if not recipients:
            print("WARNING: No alert recipients configured")
            return False
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"🚨 UNAUTHORIZED ACCESS ATTEMPT - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Prepare HTML content
            html_body = self._create_alert_html(
                timestamp=timestamp,
                similarity_score=similarity_score,
                best_match_name=best_match_name,
                has_image=image_path is not None
            )
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach image if provided
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=Path(image_path).name)
                    image.add_header('Content-ID', '<captured_image>')
                    msg.attach(image)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✓ Alert email sent to {len(recipients)} recipient(s)")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to send alert email: {e}")
            return False
    
    def _create_alert_html(self, 
                          timestamp: datetime,
                          similarity_score: float,
                          best_match_name: str = None,
                          has_image: bool = False) -> str:
        """Create HTML email body from template"""
        
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        except Exception as e:
            print(f"ERROR: Could not load email template: {e}")
            return "<h1>CRITICAL ALERTS ERROR: Template not found</h1>"

        # Prepare formatting variables
        closest_match_section = ""
        if best_match_name:
            closest_match_section = f"""
            <div class="detail-row">
                <span class="detail-label">Closest Match:</span>
                <span class="detail-value">{best_match_name}</span>
            </div>
            """
            
        image_section = ""
        if has_image:
            image_section = """
            <div class="image-container">
                <h3>Captured Image</h3>
                <img src="cid:captured_image" alt="Captured facial image">
            </div>
            """

        # Perform simple string replacement (simulating Jinja2 for minimal dependencies)
        return template.format(
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            similarity_score=f"{similarity_score:.4f}",
            system_id=config.system_id,
            vehicle_plate=config.vehicle_plate,
            owner_name=config.owner_name,
            verification_threshold=f"{config.verification_threshold:.4f}",
            closest_match_section=closest_match_section,
            image_section=image_section
        )

    def send_test_email(self, recipient: str) -> bool:
        """
        Send a test email to verify configuration
        """
        if not self.is_configured:
            print("ERROR: Email not configured")
            return False
        
        try:
            msg = MIMEText("This is a test email from the Driver Verification System. Email configuration is working correctly!")
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = recipient
            msg['Subject'] = "Test Email - Driver Verification System"
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✓ Test email sent to {recipient}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to send test email: {e}")
            return False


if __name__ == "__main__":
    # Test email service
    print("Testing email service...")
    
    service = EmailService()
    
    if service.is_configured:
        print("\nEmail service is configured")
        print(f"SMTP Server: {service.smtp_server}:{service.smtp_port}")
        print(f"Sender: {service.sender_email}")
    else:
        print("\nEmail service is NOT configured")
        print("Please update config/.env with your SMTP credentials")
