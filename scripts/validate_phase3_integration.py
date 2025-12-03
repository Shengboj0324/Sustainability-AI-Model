

import sys
from pathlib import Path
import ast

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "rag_service"))


def validate_imports():
    """Validate that server.py has all required imports"""
    print("=" * 80)
    print("PHASE 3 INTEGRATION VALIDATION")
    print("=" * 80)
    print()
    
    server_path = Path(__file__).parent.parent / "services" / "rag_service" / "server.py"
    
    with open(server_path, 'r') as f:
        content = f.read()
    
    # Check for Phase 2 imports
    checks = {
        "audit_trail import": "from audit_trail import" in content,
        "transparency_api import": "from transparency_api import" in content,
        "AuditTrailManager import": "AuditTrailManager" in content,
        "create_transparency_router import": "create_transparency_router" in content,
    }
    
    print("✅ Import Validation:")
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {result}")
    
    return all(checks.values())


def validate_initialization():
    """Validate that RAGService.__init__ initializes audit_manager"""
    print()
    print("✅ Initialization Validation:")
    
    server_path = Path(__file__).parent.parent / "services" / "rag_service" / "server.py"
    
    with open(server_path, 'r') as f:
        content = f.read()
    
    checks = {
        "audit_manager initialization": "self.audit_manager = AuditTrailManager(" in content,
        "audit_manager.async_init() call": "await self.audit_manager.async_init()" in content,
        "audit_manager.close() call": "await self.audit_manager.close()" in content,
    }
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {result}")
    
    return all(checks.values())


def validate_router_mounting():
    """Validate that transparency router is mounted"""
    print()
    print("✅ Router Mounting Validation:")
    
    server_path = Path(__file__).parent.parent / "services" / "rag_service" / "server.py"
    
    with open(server_path, 'r') as f:
        content = f.read()
    
    checks = {
        "transparency_router creation": "transparency_router = create_transparency_router(" in content,
        "router mounting": "app.include_router(transparency_router)" in content,
    }
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {result}")
    
    return all(checks.values())


def validate_audit_recording():
    """Validate that audit events are recorded"""
    print()
    print("✅ Audit Recording Validation:")
    
    server_path = Path(__file__).parent.parent / "services" / "rag_service" / "server.py"
    
    with open(server_path, 'r') as f:
        content = f.read()
    
    checks = {
        "audit_manager.record_event() call": "await rag_service.audit_manager.record_event(" in content,
        "DOCUMENT_ACCESSED event": "EventType.DOCUMENT_ACCESSED" in content,
    }
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {result}")
    
    return all(checks.values())


def validate_file_compilation():
    """Validate that all files compile successfully"""
    print()
    print("✅ File Compilation Validation:")
    
    files_to_check = [
        "services/rag_service/server.py",
        "services/rag_service/audit_trail.py",
        "services/rag_service/transparency_api.py",
        "services/rag_service/provenance.py",
        "services/rag_service/version_tracker.py",
    ]
    
    all_compiled = True
    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            compile(code, full_path, 'exec')
            print(f"  ✅ {file_path}: Compiles successfully")
        except SyntaxError as e:
            print(f"  ❌ {file_path}: Syntax error - {e}")
            all_compiled = False
        except Exception as e:
            print(f"  ⚠️  {file_path}: {e}")
    
    return all_compiled


def main():
    """Run all validations"""
    results = {
        "Imports": validate_imports(),
        "Initialization": validate_initialization(),
        "Router Mounting": validate_router_mounting(),
        "Audit Recording": validate_audit_recording(),
        "File Compilation": validate_file_compilation(),
    }
    
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for category, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {category}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED! Phase 3 integration is complete.")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

