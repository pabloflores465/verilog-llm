"""
Verilog syntax and functional validation using Icarus Verilog.
Ensures every training example compiles before inclusion.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional


class VerilogValidator:
    """Validates Verilog code using Icarus Verilog (iverilog)."""
    
    def __init__(self, iverilog_path: str = "iverilog"):
        self.iverilog = iverilog_path
        self._check_iverilog()
    
    def _check_iverilog(self):
        result = subprocess.run([self.iverilog, "-V"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("iverilog not found. Install: brew install icarus-verilog")
    
    def check_syntax(self, code: str, module_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if Verilog code compiles with iverilog.
        Returns (is_valid, error_message).
        """
        # Clean up code
        code = code.strip()
        if not code:
            return False, "Empty code"
        
        # Wrap raw code in module if needed for standalone compilation
        test_code = self._prepare_for_compile(code, module_name)
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                vfile = Path(tmpdir) / "test.v"
                vfile.write_text(test_code)
                
                # Compile with iverilog (syntax check only, no elaboration needed)
                result = subprocess.run(
                    [self.iverilog, "-g2012", "-o", str(Path(tmpdir) / "test.out"), str(vfile)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode == 0:
                    return True, ""
                
                # Parse error message
                err = result.stderr or result.stdout
                # Clean up temp paths from error
                err = err.replace(str(tmpdir), "").replace(str(vfile), "test.v")
                return False, err
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout (>30s)"
        except Exception as e:
            return False, f"Validator error: {str(e)}"
    
    def _prepare_for_compile(self, code: str, module_name: Optional[str] = None) -> str:
        """
        Ensure code is compilable standalone.
        If the code has no module/endmodule, wrap it.
        """
        if "module" in code and "endmodule" in code:
            return code
        
        # If it's just body code, wrap in a test module
        mod_name = module_name or "test_module"
        return f"""module {mod_name} ();
{code}
endmodule
"""
    
    def quick_check(self, code: str) -> bool:
        """Quick boolean syntax check."""
        ok, _ = self.check_syntax(code)
        return ok
    
    def extract_module_name(self, code: str) -> Optional[str]:
        """Extract module name from Verilog code."""
        match = re.search(r'module\s+(\w+)', code)
        return match.group(1) if match else None
    
    def validate_dataset(self, examples: list, verbose: bool = True) -> Tuple[list, list]:
        """
        Validate a list of examples. Returns (valid_examples, invalid_examples).
        Each example should have a 'code' field.
        """
        valid = []
        invalid = []
        
        for i, ex in enumerate(examples):
            code = ex.get("code", "")
            mod_name = self.extract_module_name(code)
            is_valid, err = self.check_syntax(code, mod_name)
            
            if is_valid:
                valid.append(ex)
            else:
                ex["_error"] = err
                invalid.append(ex)
                
            if verbose and (i + 1) % 100 == 0:
                print(f"  Validated {i+1}/{len(examples)}: {len(valid)} valid, {len(invalid)} invalid")
        
        if verbose:
            total = len(examples)
            print(f"\nValidation complete: {len(valid)}/{total} valid ({len(valid)/total*100:.1f}%)")
        
        return valid, invalid
