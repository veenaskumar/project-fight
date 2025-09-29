#!/usr/bin/env python3
"""
Code validation script for Violence Detection System
Checks if all components can be imported and basic functionality works
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'streamlit',
        'fastapi',
        'uvicorn',
        'cv2',
        'numpy',
        'requests',
        'boto3',
        'ultralytics',
        'websocket',
        'twilio',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'websocket':
                import websocket
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_file_syntax():
    """Test if Python files have valid syntax"""
    print("\n🔍 Testing file syntax...")
    
    python_files = [
        'CPU_Server.py',
        'GUP_server.py', 
        'index.py'
    ]
    
    syntax_errors = []
    
    for file in python_files:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    compile(f.read(), file, 'exec')
                print(f"  ✅ {file}")
            except SyntaxError as e:
                print(f"  ❌ {file}: {e}")
                syntax_errors.append(file)
        else:
            print(f"  ⚠️  {file}: File not found")
    
    if syntax_errors:
        print(f"\n❌ Syntax errors in: {', '.join(syntax_errors)}")
        return False
    else:
        print("\n✅ All files have valid syntax!")
        return True

def test_environment():
    """Test environment configuration"""
    print("\n🔍 Testing environment...")
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("  ✅ .env file exists")
    else:
        print("  ⚠️  .env file not found (create from env.example)")
    
    # Check if model file exists
    if os.path.exists('violence_detection_v4.pt'):
        print("  ✅ Model file exists")
    else:
        print("  ❌ violence_detection_v4.pt not found")
        return False
    
    # Check if logo exists
    if os.path.exists('logo.png'):
        print("  ✅ Logo file exists")
    else:
        print("  ⚠️  logo.png not found")
    
    # Check if styles exists
    if os.path.exists('styles.css'):
        print("  ✅ Styles file exists")
    else:
        print("  ⚠️  styles.css not found")
    
    return True

def test_cpu_server():
    """Test CPU server basic functionality"""
    print("\n🔍 Testing CPU server...")
    
    try:
        # Test imports
        import CPU_Server
        print("  ✅ CPU_Server.py imports successfully")
        
        # Test basic functions
        streams = CPU_Server.load_streams_from_s3()
        print("  ✅ load_streams_from_s3() works")
        
        logs = CPU_Server.load_logs_from_s3()
        print("  ✅ load_logs_from_s3() works")
        
        return True
    except Exception as e:
        print(f"  ❌ CPU server error: {e}")
        return False

def test_gpu_server():
    """Test GPU server basic functionality"""
    print("\n🔍 Testing GPU server...")
    
    try:
        # Test imports
        import GUP_server
        print("  ✅ GUP_server.py imports successfully")
        
        # Test basic functions
        class_name = GUP_server.get_class_name(0)
        print(f"  ✅ get_class_name() works: {class_name}")
        
        return True
    except Exception as e:
        print(f"  ❌ GPU server error: {e}")
        return False

def test_frontend():
    """Test frontend basic functionality"""
    print("\n🔍 Testing frontend...")
    
    try:
        # Test imports
        import index
        print("  ✅ index.py imports successfully")
        
        # Test configuration
        if hasattr(index, 'MJPEG_URL'):
            print("  ✅ MJPEG_URL defined")
        else:
            print("  ❌ MJPEG_URL not defined")
            return False
        
        if hasattr(index, 'CPU_SERVICE_URL'):
            print("  ✅ CPU_SERVICE_URL defined")
        else:
            print("  ❌ CPU_SERVICE_URL not defined")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Frontend error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Violence Detection System - Code Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_file_syntax,
        test_environment,
        test_cpu_server,
        test_gpu_server,
        test_frontend
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Code should run correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
