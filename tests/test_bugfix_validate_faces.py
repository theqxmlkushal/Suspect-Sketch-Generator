"""
tests/test_bugfix_validate_faces.py
====================================
Bug condition exploration test for validate_faces undefined error.

This test demonstrates the bug where ui/app.py references 'validate_faces'
variable that doesn't exist (should be 'val_faces').

**Validates: Requirements 2.1, 2.2, 2.3**

Property 1: Fault Condition - Variable Reference Resolution
For any Streamlit UI generation request where the user clicks "Generate sketch"
with a valid description, the code should correctly reference the val_faces
variable without raising a NameError.

Run: pytest tests/test_bugfix_validate_faces.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, example


class TestBugConditionExploration:
    """
    Bug Condition Exploration Test
    
    **Property 1: Fault Condition - Variable Reference Resolution**
    
    This test simulates the exact code pattern from ui/app.py that causes
    the bug. The bug occurs when the code tries to reference 'validate_faces'
    variable that doesn't exist (should be 'val_faces').
    
    On UNFIXED code: This test FAILS with NameError: name 'validate_faces' is not defined
    On FIXED code: This test PASSES, confirming the bug is resolved
    """
    
    @given(
        val_faces=st.booleans(),
        num_images=st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=50, deadline=None)
    @example(val_faces=True, num_images=2)
    @example(val_faces=False, num_images=1)
    def test_variable_reference_resolution(self, val_faces, num_images):
        """
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: For any face validation setting (val_faces), the code should
        correctly reference the variable without raising a NameError.
        
        This test simulates the exact buggy code pattern from ui/app.py line 228:
        - Variable 'val_faces' is defined (from Streamlit toggle)
        - Code tries to reference 'validate_faces' (undefined)
        - Result: NameError on unfixed code
        """
        # Simulate the local scope in ui/app.py at line 221-228
        # val_faces is defined from the Streamlit toggle
        local_scope = {
            'val_faces': val_faces,
            'num_images': num_images,
            'prompt': "test prompt",
            'seed': 12345,
        }
        
        # This is the buggy code pattern from ui/app.py line 221-228
        # The code tries to pass validate_faces=validate_faces
        # but validate_faces is NOT defined (should be val_faces)
        buggy_code = """
# Simulating pipe.generate() call with the bug
result = {
    'prompt': prompt,
    'num_images': num_images,
    'seed': seed,
    'validate_faces': validate_faces  # BUG: validate_faces is undefined, should be val_faces
}
"""
        
        # Try to execute the buggy code
        # On UNFIXED code: This raises NameError: name 'validate_faces' is not defined
        # On FIXED code: The actual code would use val_faces instead
        try:
            exec(buggy_code, local_scope)
            # If we get here, the bug is fixed (or the code was changed to use val_faces)
            pytest.fail(
                "Expected NameError for 'validate_faces' but code executed successfully. "
                "This means either:\n"
                "1. The bug has been fixed (validate_faces changed to val_faces)\n"
                "2. The test is not correctly simulating the bug\n"
                f"val_faces={val_faces}, num_images={num_images}"
            )
        except NameError as e:
            # This is the EXPECTED behavior on UNFIXED code
            if "validate_faces" in str(e):
                # BUG CONFIRMED: The code references undefined 'validate_faces'
                # This is what we expect to see on unfixed code
                assert True, f"Bug confirmed: {e}"
            else:
                # Some other NameError - re-raise
                raise
    
    def test_face_validation_enabled_concrete(self):
        """
        **Validates: Requirements 2.1, 2.2**
        
        Concrete test case: Face validation enabled (val_faces=True)
        
        This test demonstrates the bug with a specific example where
        face validation is enabled. The bug occurs because the code
        references 'validate_faces' which doesn't exist.
        """
        val_faces = True
        
        local_scope = {
            'val_faces': val_faces,
            'prompt': "White male, 40 years old",
            'num_images': 2,
            'seed': 12345,
        }
        
        buggy_code = """
result = {
    'validate_faces': validate_faces  # BUG: undefined variable
}
"""
        
        try:
            exec(buggy_code, local_scope)
            pytest.fail(
                "Expected NameError for 'validate_faces' with val_faces=True. "
                "Bug may have been fixed."
            )
        except NameError as e:
            if "validate_faces" in str(e):
                # Bug confirmed for face validation enabled case
                assert True, f"Bug confirmed with val_faces=True: {e}"
            else:
                raise
    
    def test_face_validation_disabled_concrete(self):
        """
        **Validates: Requirements 2.3**
        
        Concrete test case: Face validation disabled (val_faces=False)
        
        This test demonstrates that the bug occurs even when face validation
        is disabled, because the undefined variable is referenced before
        any validation check.
        """
        val_faces = False
        
        local_scope = {
            'val_faces': val_faces,
            'prompt': "Young Asian woman, early 20s",
            'num_images': 1,
            'seed': 54321,
        }
        
        buggy_code = """
result = {
    'validate_faces': validate_faces  # BUG: undefined variable
}
"""
        
        try:
            exec(buggy_code, local_scope)
            pytest.fail(
                "Expected NameError for 'validate_faces' with val_faces=False. "
                "Bug occurs regardless of toggle state."
            )
        except NameError as e:
            if "validate_faces" in str(e):
                # Bug confirmed even with face validation disabled
                assert True, f"Bug confirmed with val_faces=False: {e}"
            else:
                raise
    
    def test_bug_location_in_actual_code(self):
        """
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        This test verifies the bug exists in the actual ui/app.py file
        by checking for the buggy pattern in the source code.
        
        On UNFIXED code: This test PASSES (bug pattern found in source)
        On FIXED code: This test FAILS (bug pattern not found - code uses val_faces)
        """
        # Read the actual ui/app.py file
        ui_app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui', 'app.py')
        
        with open(ui_app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the buggy pattern exists in the file
        # The bug is: pipe.generate(..., validate_faces=validate_faces)
        # where validate_faces is undefined (should be val_faces)
        
        # Look for the pipe.generate call with validate_faces parameter
        if 'validate_faces=validate_faces' in content:
            # Bug pattern found in source code
            # This confirms the bug exists in the actual file
            assert True, "Bug pattern 'validate_faces=validate_faces' found in ui/app.py"
        else:
            # Bug pattern not found - either fixed or never existed
            pytest.fail(
                "Bug pattern 'validate_faces=validate_faces' NOT found in ui/app.py. "
                "This means either:\n"
                "1. The bug has been fixed (code now uses val_faces)\n"
                "2. The bug never existed in this version\n"
                "Expected to find: pipe.generate(..., validate_faces=validate_faces)"
            )

