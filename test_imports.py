import sys
sys.path.insert(0, 'src')

print("Testing Driver model...")
from database.models import Driver
d = Driver(name='Test', category='B', license_number='LIC-001')
info = d.to_dict()
assert info['category'] == 'B', "category missing from to_dict!"
assert info['license_number'] == 'LIC-001', "license_number missing from to_dict!"
print("  OK:", info)

print("\nTesting repository imports...")
from database.driver_repository import DriverRepository
from database.verification_repository import VerificationRepository
from database.audit_repository import AuditRepository
from database.db_manager import DatabaseManager
print("  OK: all repository classes imported")

print("\nTesting config.database_url...")
from utils.config import config
url = config.database_url
assert url.startswith("postgresql://"), f"Expected postgresql:// URL, got: {url}"
print("  OK:", url)

print("\n✅ All checks passed! Ready for PostgreSQL connection.")
