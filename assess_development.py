#!/usr/bin/env python3
"""
Comprehensive VishwamAI Development and Testing Assessment

This script runs all tests and provides a detailed assessment of the 
VishwamAI model development status, including distillation functionality.
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add VishwamAI to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_command(cmd, cwd=None):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        'jax': False,
        'torch': False,
        'transformers': False,
        'duckdb': False,
        'pandas': False,
        'numpy': False,
        'pytest': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - not available")
    
    return dependencies

def test_core_imports():
    """Test core VishwamAI imports."""
    print("\n🧪 Testing Core Imports...")
    
    tests = {
        'vishwamai.model': False,
        'vishwamai.distillation': False,
        'vishwamai.attention': False,
        'vishwamai.layers': False,
        'vishwamai.training': False,
        'vishwamai.utils': False
    }
    
    for module in tests:
        try:
            __import__(module)
            tests[module] = True
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {e}")
        except Exception as e:
            print(f"  ⚠️  {module} - {e}")
    
    return tests

def test_distillation_functionality():
    """Test distillation functionality specifically."""
    print("\n🔬 Testing Distillation Functionality...")
    
    results = {
        'config_creation': False,
        'tracker_creation': False,
        'no_wandb_refs': False,
        'syntax_valid': False
    }
    
    # Test syntax validity
    try:
        import py_compile
        distillation_path = project_root / "vishwamai" / "distillation.py"
        py_compile.compile(str(distillation_path), doraise=True)
        results['syntax_valid'] = True
        print("  ✅ Distillation syntax valid")
    except Exception as e:
        print(f"  ❌ Distillation syntax error: {e}")
    
    # Test config creation
    try:
        from vishwamai.distillation import DistillationConfig
        config = DistillationConfig(use_duckdb_tracking=True)
        results['config_creation'] = True
        print("  ✅ DistillationConfig creation")
    except Exception as e:
        print(f"  ❌ DistillationConfig creation failed: {e}")
    
    # Test tracker creation
    try:
        from vishwamai.distillation import DuckDBDistillationTracker
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            tracker = DuckDBDistillationTracker(tmp.name, "test")
            tracker.close()
        results['tracker_creation'] = True
        print("  ✅ DuckDBDistillationTracker creation")
    except Exception as e:
        print(f"  ❌ DuckDBDistillationTracker creation failed: {e}")
    
    # Test no wandb references
    try:
        distillation_path = project_root / "vishwamai" / "distillation.py"
        with open(distillation_path, 'r') as f:
            content = f.read()
        
        wandb_imports = [line for line in content.split('\n') 
                        if ('import wandb' in line or 'from wandb' in line) 
                        and not line.strip().startswith('#')]
        
        if len(wandb_imports) == 0:
            results['no_wandb_refs'] = True
            print("  ✅ No wandb imports found")
        else:
            print(f"  ❌ Found wandb imports: {wandb_imports}")
    except Exception as e:
        print(f"  ❌ Error checking wandb references: {e}")
    
    return results

def run_pytest_tests():
    """Run pytest tests if available."""
    print("\n🧪 Running Pytest Tests...")
    
    if not Path("tests").exists():
        print("  ⚠️  No tests directory found")
        return False, "No tests directory"
    
    success, stdout, stderr = run_command("python3 -m pytest tests/ -v", cwd=project_root)
    
    if success:
        print("  ✅ All pytest tests passed")
        return True, stdout
    else:
        print("  ❌ Some pytest tests failed")
        return False, stderr

def check_config_files():
    """Check configuration files."""
    print("\n📋 Checking Configuration Files...")
    
    config_checks = {
        'distillation_config.json': False,
        'valid_json': False,
        'has_duckdb_config': False
    }
    
    config_path = project_root / "configs" / "distillation_config.json"
    
    if config_path.exists():
        config_checks['distillation_config.json'] = True
        print("  ✅ distillation_config.json exists")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            config_checks['valid_json'] = True
            print("  ✅ distillation_config.json is valid JSON")
            
            if 'tracking_config' in config and 'use_duckdb' in config['tracking_config']:
                config_checks['has_duckdb_config'] = True
                print("  ✅ DuckDB configuration present")
            else:
                print("  ❌ DuckDB configuration missing")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
        except Exception as e:
            print(f"  ❌ Error reading config: {e}")
    else:
        print("  ❌ distillation_config.json not found")
    
    return config_checks

def assess_model_completeness():
    """Assess completeness of model components."""
    print("\n🏗️  Assessing Model Completeness...")
    
    required_files = {
        'vishwamai/__init__.py': Path("vishwamai/__init__.py"),
        'vishwamai/model.py': Path("vishwamai/model.py"),
        'vishwamai/distillation.py': Path("vishwamai/distillation.py"),
        'vishwamai/attention.py': Path("vishwamai/attention.py"),
        'vishwamai/layers.py': Path("vishwamai/layers.py"),
        'vishwamai/training.py': Path("vishwamai/training.py"),
        'requirements.txt': Path("requirements.txt"),
        'setup.py': Path("setup.py")
    }
    
    completeness = {}
    
    for name, path in required_files.items():
        if path.exists() and path.stat().st_size > 0:
            completeness[name] = True
            print(f"  ✅ {name}")
        else:
            completeness[name] = False
            print(f"  ❌ {name} missing or empty")
    
    return completeness

def generate_report():
    """Generate comprehensive development assessment report."""
    print("\n" + "="*60)
    print("🎯 VISHWAMAI DEVELOPMENT ASSESSMENT REPORT")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'assessment_results': {}
    }
    
    # Run all assessments
    dependencies = check_dependencies()
    imports = test_core_imports()
    distillation = test_distillation_functionality()
    configs = check_config_files()
    completeness = assess_model_completeness()
    
    # Try running pytest
    pytest_success, pytest_output = run_pytest_tests()
    
    # Store results
    report['assessment_results'] = {
        'dependencies': dependencies,
        'core_imports': imports,
        'distillation_functionality': distillation,
        'config_files': configs,
        'model_completeness': completeness,
        'pytest_tests': {
            'success': pytest_success,
            'output_preview': pytest_output[:500] if pytest_output else "No output"
        }
    }
    
    # Calculate scores
    dep_score = sum(dependencies.values()) / len(dependencies) * 100
    import_score = sum(imports.values()) / len(imports) * 100
    distill_score = sum(distillation.values()) / len(distillation) * 100
    config_score = sum(configs.values()) / len(configs) * 100
    complete_score = sum(completeness.values()) / len(completeness) * 100
    
    overall_score = (dep_score + import_score + distill_score + config_score + complete_score) / 5
    
    print(f"\n📊 ASSESSMENT SCORES:")
    print(f"  Dependencies Available: {dep_score:.1f}%")
    print(f"  Core Imports Working: {import_score:.1f}%")
    print(f"  Distillation Functionality: {distill_score:.1f}%")
    print(f"  Configuration Files: {config_score:.1f}%")
    print(f"  Model Completeness: {complete_score:.1f}%")
    print(f"  Pytest Tests: {'✅ PASS' if pytest_success else '❌ FAIL'}")
    print(f"\n🎯 OVERALL SCORE: {overall_score:.1f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if dep_score < 80:
        print("  📦 Install missing dependencies with: pip install -r requirements.txt")
    
    if import_score < 80:
        print("  🔧 Fix import issues in core modules")
    
    if distill_score < 100:
        print("  🧠 Complete distillation module implementation")
        
    if config_score < 100:
        print("  ⚙️  Fix configuration files")
        
    if complete_score < 90:
        print("  📁 Add missing core files")
    
    if not pytest_success:
        print("  🧪 Fix failing tests")
    
    if overall_score >= 90:
        print("  🎉 Excellent! VishwamAI is development-ready!")
    elif overall_score >= 70:
        print("  👍 Good progress! Minor fixes needed for production readiness")
    elif overall_score >= 50:
        print("  ⚠️  Moderate issues. Address key dependencies and imports")
    else:
        print("  🚨 Major issues detected. Focus on basic setup first")
    
    # Save report
    report_path = project_root / "development_assessment_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: {report_path}")
    
    return overall_score, report

def main():
    """Main assessment function."""
    print("🚀 VishwamAI Development Assessment")
    print("=" * 40)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project Root: {project_root}")
    
    try:
        score, report = generate_report()
        
        print(f"\n✨ Assessment completed with score: {score:.1f}%")
        
        if score >= 80:
            print("🎊 VishwamAI is ready for development and testing!")
            return 0
        else:
            print("🔧 Some issues need attention before development can proceed smoothly.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Assessment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
