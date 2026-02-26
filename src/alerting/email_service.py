"""
Email Service Module
Handles SMTP email alerts for unauthorized access attempts
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import List, Optional
from utils.config import config

# ---------------------------------------------------------------------------
# Email HTML template (inlined — no external file dependency)
# Uses str.format() placeholders: {timestamp}, {similarity_score}, etc.
# ---------------------------------------------------------------------------
_EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
  .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
  .header {{ background-color: #dc3545; color: white; padding: 20px;
             text-align: center; border-radius: 5px 5px 0 0; }}
  .warning-icon {{ font-size: 48px; margin-bottom: 10px; }}
  .content {{ background-color: #f8f9fa; padding: 20px;
              border: 1px solid #dee2e6; }}
  .detail-row {{ margin: 10px 0; padding: 10px; background-color: white;
                 border-left: 4px solid #dc3545; }}
  .detail-label {{ font-weight: bold; color: #495057; }}
  .detail-value  {{ color: #212529; }}
  .image-container {{ margin: 20px 0; text-align: center; }}
  .image-container img {{ max-width: 100%; border: 2px solid #dc3545;
                          border-radius: 5px; }}
  .footer {{ margin-top: 20px; padding: 15px; background-color: #e9ecef;
             text-align: center; font-size: 12px; color: #6c757d;
             border-radius: 0 0 5px 5px; }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="warning-icon">⚠️</div>
      <h1>UNAUTHORIZED ACCESS ATTEMPT</h1>
      <p>Driver Identity Verification System</p>
    </div>
    <div class="content">
      <h2>Alert Details</h2>
      <div class="detail-row">
        <span class="detail-label">Timestamp:</span>
        <span class="detail-value">{timestamp}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Status:</span>
        <span class="detail-value" style="color:#dc3545;font-weight:bold;">UNAUTHORIZED</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Similarity Score:</span>
        <span class="detail-value">{similarity_score}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">System ID:</span>
        <span class="detail-value">{system_id}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Vehicle Plate:</span>
        <span class="detail-value">{vehicle_plate}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Vehicle Owner:</span>
        <span class="detail-value">{owner_name}</span>
      </div>
      {closest_match_section}
      <div class="detail-row">
        <span class="detail-label">Verification Threshold:</span>
        <span class="detail-value">{verification_threshold}</span>
      </div>
      {image_section}
      <div style="margin-top:20px;padding:15px;background-color:#fff3cd;
                  border-left:4px solid #ffc107;">
        <strong>⚠️ Action Required:</strong>
        <p>An unauthorized individual attempted to access the system.
           Please review the captured image and take appropriate action.</p>
      </div>
    </div>
    <div class="footer">
      <p>This is an automated alert from the Driver Identity Verification System.</p>
      <p>Please do not reply to this email.</p>
    </div>
  </div>
</body>
</html>"""


class EmailService:
    """Manages email alerts for security events"""

    def __init__(self):
        self.smtp_server     = config.smtp_server
        self.smtp_port       = config.smtp_port
        self.sender_email    = config.smtp_email
        self.sender_password = config.smtp_password
        self.sender_name     = config.get('email.sender_name', 'Driver Verification System')
        self.is_configured   = self._check_configuration()

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
        """Build HTML email body from the inlined template constant."""
        closest_match_section = ""
        if best_match_name:
            closest_match_section = f"""
      <div class="detail-row">
        <span class="detail-label">Closest Match:</span>
        <span class="detail-value">{best_match_name}</span>
      </div>"""

        image_section = ""
        if has_image:
            image_section = """
      <div class="image-container">
        <h3>Captured Image</h3>
        <img src="cid:captured_image" alt="Captured facial image">
      </div>"""

        return _EMAIL_TEMPLATE.format(
            timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            similarity_score=f"{similarity_score:.4f}",
            system_id=config.system_id,
            vehicle_plate=config.vehicle_plate,
            owner_name=config.owner_name,
            verification_threshold=f"{config.verification_threshold:.4f}",
            closest_match_section=closest_match_section,
            image_section=image_section,
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
