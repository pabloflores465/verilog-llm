"""
Verilog post-processor: fixes common LLM generation errors.
Critical for begin/end, module/endmodule matching, and basic syntax repair.
"""

import re
from typing import Tuple, Optional


class VerilogPostProcessor:
    """
    Repairs common Verilog generation mistakes from LLMs.
    Inspired by HaVen's code extractor + additional heuristic fixes.
    """

    def __init__(self):
        self.keywords_needing_begin = [
            'always', 'always_ff', 'always_comb', 'always_latch',
            'if', 'else', 'for', 'while', 'case', 'casex', 'casez', 'forever', 'repeat',
        ]
        self.block_keywords = ['begin', 'end', 'fork', 'join', 'join_any', 'join_none']

    def process(self, raw_text: str, expected_module_name: Optional[str] = None) -> str:
        """Full pipeline: extract -> clean -> fix -> validate."""
        code = self._extract_code_block(raw_text)
        code = self._normalize_whitespace(code)
        code = self._ensure_module_wrapper(code, expected_module_name)
        code = self._fix_begin_end_balance(code)
        code = self._fix_parentheses_balance(code)
        code = self._remove_duplicate_endmodule(code)
        code = self._fix_semicolons(code)
        code = self._fix_indentation(code)
        return code.strip()

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown or raw text."""
        # Try explicit verilog blocks
        for pat in [r'```verilog\s*(.*?)```', r'```\s*(.*?)```', r'<answer>\s*(.*?)\s*</answer>']:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Find module/endmodule span
        mod_start = text.find('module ')
        endmod_pos = text.rfind('endmodule')
        if mod_start != -1 and endmod_pos != -1 and endmod_pos > mod_start:
            return text[mod_start:endmod_pos + len('endmodule')]

        # Fallback: return everything (will be wrapped later)
        return text.strip()

    def _normalize_whitespace(self, code: str) -> str:
        """Clean up whitespace and remove thinking artifacts."""
        # Remove think blocks if leaked into code
        code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)
        code = re.sub(r'</?think>', '', code)
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        return code

    def _ensure_module_wrapper(self, code: str, expected_name: Optional[str] = None) -> str:
        """If code lacks module/endmodule, wrap it."""
        has_module = re.search(r'\bmodule\s+\w+', code) is not None
        has_endmodule = 'endmodule' in code

        if has_module and has_endmodule:
            return code

        mod_name = expected_name or 'generated_module'

        # If it has module but no endmodule, add one
        if has_module and not has_endmodule:
            return code + '\nendmodule\n'

        # If it has neither, wrap entire body
        if not has_module and not has_endmodule:
            ports = self._infer_ports(code)
            header = f"module {mod_name} (\n{ports}\n);"
            return header + '\n' + code + '\nendmodule\n'

        # Has endmodule but no module header (weird but possible)
        if not has_module and has_endmodule:
            ports = self._infer_ports(code)
            header = f"module {mod_name} (\n{ports}\n);"
            return header + '\n' + code

        return code

    def _infer_ports(self, code: str) -> str:
        """Infer port list from input/output declarations in body."""
        ports = []
        for m in re.finditer(r'\b(input|output|inout)\s+(?:reg\s+|wire\s+)?(?:\[.*?\]\s+)?(\w+)', code):
            ports.append(f'    {m.group(2)}')
        return ',\n'.join(ports) if ports else '    // no ports inferred'

    def _fix_begin_end_balance(self, code: str) -> str:
        """
        Heuristic fix for mismatched begin/end.
        This is the #1 source of LLM Verilog syntax errors.
        """
        lines = code.split('\n')
        stack = []       # stack of (line_idx, keyword)
        inserts = []     # lines to insert after
        deletes = set()  # lines to delete

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue

            # Count begin/end on this line
            begins = stripped.count('begin')
            # Don't count 'begin' inside strings/comments (simplified)
            ends = stripped.count('end')
            # Avoid counting 'end' inside 'endmodule', 'endcase', 'endgenerate', 'endfunction'
            special_ends = len(re.findall(r'\b(endmodule|endcase|endgenerate|endfunction|endtask)\b', stripped))
            net_ends = ends - special_ends

            # Push begins
            for _ in range(begins):
                stack.append((i, 'begin'))

            # Pop ends
            for _ in range(net_ends):
                if stack:
                    stack.pop()
                else:
                    # Extra end: mark for deletion
                    deletes.add(i)

        # Missing ends: add them before the enclosing construct closes
        for line_idx, kw in reversed(stack):
            # Insert 'end' before module/endmodule, always block end, or end of file
            inserted = False
            for j in range(line_idx + 1, len(lines)):
                if re.search(r'\b(endmodule|endcase|endgenerate|endfunction|endtask)\b', lines[j]):
                    inserts.append((j, 'end'))
                    inserted = True
                    break
            if not inserted:
                inserts.append((len(lines), 'end'))

        # Apply inserts (in reverse order to preserve indices)
        inserts.sort(key=lambda x: x[0], reverse=True)
        for idx, text in inserts:
            lines.insert(idx, '    ' + text)

        # Apply deletes (in reverse order)
        for idx in sorted(deletes, reverse=True):
            del lines[idx]

        return '\n'.join(lines)

    def _fix_parentheses_balance(self, code: str) -> str:
        """Fix unbalanced parentheses in module port list and expressions."""
        # Only fix module header parentheses, not everything
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\s*module\s+\w+\s*\(', line):
                # Count parens in header and next few lines until );
                header_lines = []
                j = i
                while j < len(lines) and ');' not in lines[j]:
                    header_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    header_lines.append(lines[j])

                header_text = '\n'.join(header_lines)
                open_p = header_text.count('(')
                close_p = header_text.count(')')

                if open_p > close_p:
                    # Add missing )
                    last = header_lines[-1]
                    if not last.rstrip().endswith(')'):
                        if last.rstrip().endswith(';'):
                            header_lines[-1] = last.rstrip()[:-1] + ');'
                        else:
                            header_lines[-1] = last.rstrip() + ');'

                    for k, new_line in enumerate(header_lines):
                        lines[i + k] = new_line
                break

        return '\n'.join(lines)

    def _remove_duplicate_endmodule(self, code: str) -> str:
        """Keep only the last endmodule."""
        lines = code.split('\n')
        endmodule_indices = [i for i, l in enumerate(lines) if l.strip() == 'endmodule']

        if len(endmodule_indices) <= 1:
            return code

        # Remove all but the last endmodule
        for idx in reversed(endmodule_indices[:-1]):
            del lines[idx]

        return '\n'.join(lines)

    def _fix_semicolons(self, code: str) -> str:
        """Add missing semicolons after assignments (conservative)."""
        lines = code.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            # Skip comments, directives, begin/end, module statements, port declarations
            if (not stripped or stripped.startswith('//') or stripped.startswith('`') or
                stripped.startswith('#') or stripped in ('begin', 'end', 'endmodule') or
                stripped.startswith('module') or stripped.startswith('end') or
                stripped.startswith('input') or stripped.startswith('output') or stripped.startswith('inout')):
                result.append(line)
                continue

            # Only fix assign statements and wire/reg declarations that are assignments
            if re.search(r'\b(assign)\b', stripped) or ( '=' in stripped and not stripped.startswith('parameter') and not stripped.startswith('localparam')):
                if not stripped.rstrip().endswith((';', ',', ')', 'begin', '{')):
                    if re.search(r'[a-zA-Z0-9_\]]\s*$', stripped):
                        line = line.rstrip() + ';'

            result.append(line)
        return '\n'.join(result)

    def _fix_indentation(self, code: str) -> str:
        """Basic indentation fix."""
        lines = code.split('\n')
        indent = 0
        result = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append('')
                continue

            # Decrease indent for end keywords
            if re.match(r'^(end|endcase|endmodule|endgenerate|endfunction|endtask|join)', stripped):
                indent = max(0, indent - 4)

            result.append(' ' * indent + stripped)

            # Increase indent for begin/case keywords
            if re.search(r'\bbegin\b', stripped) and 'end' not in stripped:
                indent += 4
            if re.search(r'\b(case|casex|casez)\b', stripped) and 'endcase' not in stripped:
                indent += 4

        return '\n'.join(result)


def postprocess_verilog(raw_text: str, expected_module_name: Optional[str] = None) -> str:
    """Convenience function."""
    processor = VerilogPostProcessor()
    return processor.process(raw_text, expected_module_name)
