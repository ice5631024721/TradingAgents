#!/usr/bin/env python3
"""
Test script to verify the get_user_selections function works correctly
with the new configuration file approach.
"""

import sys
sys.path.insert(0, '.')

# Import required modules
from cli.main import get_user_selections
from cli.models import AnalystType
import os

def test_config_loading():
    """Test that configuration loading works correctly."""
    print("Testing configuration file loading...")
    
    # Check if config.yaml exists
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml not found!")
        return False
    
    try:
        # Call the modified get_user_selections function
        selections = get_user_selections()
        
        # Verify the returned structure
        expected_keys = {
            'ticker', 'analysis_date', 'analysts', 'research_depth',
            'llm_provider', 'backend_url', 'shallow_thinker', 'deep_thinker'
        }
        
        if not all(key in selections for key in expected_keys):
            print("❌ Missing required keys in selections")
            return False
        
        # Verify analysts are AnalystType enums
        if not all(isinstance(analyst, AnalystType) for analyst in selections['analysts']):
            print("❌ Analysts are not properly converted to AnalystType enums")
            return False
        
        print("✅ Configuration loading test passed!")
        print(f"   Loaded ticker: {selections['ticker']}")
        print(f"   Loaded analysts: {[a.value for a in selections['analysts']]}")
        print(f"   Loaded research depth: {selections['research_depth']}")
        print(f"   Loaded LLM provider: {selections['llm_provider']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during configuration loading: {e}")
        return False

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        print("\n🎉 All tests passed! The configuration file approach is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the configuration.")
        sys.exit(1)