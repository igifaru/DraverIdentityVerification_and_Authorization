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
        
        self.is_configured = self._check_configuration()
        
        if self.is_configured:
            print("✓ Email service configured")
        else:
            print("⚠️  Email service not configured (alerts will not be sent)")
    
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
            
            # Create HTML body
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
        """Create HTML email body for alert"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #dc3545;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px 5px 0 0;
                }}
                .content {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border: 1px solid #dee2e6;
                }}
                .detail-row {{
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-left: 4px solid #dc3545;
                }}
                .detail-label {{
                    font-weight: bold;
                    color: #495057;
                }}
                .detail-value {{
                    color: #212529;
                }}
                .image-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .image-container img {{
                    max-width: 100%;
                    border: 2px solid #dc3545;
                    border-radius: 5px;
                }}
                .footer {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #e9ecef;
                    text-align: center;
                    font-size: 12px;
                    color: #6c757d;
                    border-radius: 0 0 5px 5px;
                }}
                .warning-icon {{
                    font-size: 48px;
                    margin-bottom: 10px;
                }}
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
                        <span class="detail-value">{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Status:</span>
                        <span class="detail-value" style="color: #dc3545; font-weight: bold;">UNAUTHORIZED</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Similarity Score:</span>
                        <span class="detail-value">{similarity_score:.4f}</span>
                    </div>
                    
                    {f'''
                    <div class="detail-row">
                        <span class="detail-label">Closest Match:</span>
                        <span class="detail-value">{best_match_name}</span>
                    </div>
                    ''' if best_match_name else ''}
                    
                    <div class="detail-row">
                        <span class="detail-label">Verification Threshold:</span>
                        <span class="detail-value">{config.verification_threshold:.4f}</span>
                    </div>
                    
                    {'''
                    <div class="image-container">
                        <h3>Captured Image</h3>
                        <img src="cid:captured_image" alt="Captured facial image">
                    </div>
                    ''' if has_image else ''}
                    
                    <div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
                        <strong>⚠️ Action Required:</strong>
                        <p>An unauthorized individual attempted to access the system. Please review the captured image and take appropriate action.</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is an automated alert from the Driver Identity Verification System.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_test_email(self, recipient: str) -> bool:
        """
        Send a test email to verify configuration
        
        Args:
            recipient: Test recipient email address
            
        Returns:
            True if successful, False otherwise
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
        
        # Uncomment to send test email
        # test_recipient = input("Enter test recipient email: ")
        # service.send_test_email(test_recipient)
    else:
        print("\nEmail service is NOT configured")
        print("Please update config/.env with your SMTP credentials")
