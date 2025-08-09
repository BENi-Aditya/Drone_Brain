#!/usr/bin/env python3
"""
Python Indentation Checker
Helps identify and fix indentation issues in Python files
"""

import sys
import ast
import tokenize
import io

def check_indentation(filename):
    """Check a Python file for indentation issues"""
    print(f"Checking indentation in: {filename}")
    print("=" * 50)
    
    try:
        # First, try to parse the file with AST
        with open(filename, 'r') as f:
            content = f.read()
        
        try:
            ast.parse(content)
            print("✅ File parses successfully - no syntax errors")
        except SyntaxError as e:
            print(f"❌ Syntax Error found:")
            print(f"   Line {e.lineno}: {e.msg}")
            if e.text:
                print(f"   Code: {e.text.strip()}")
            print()
            
            # Show the problematic area
            lines = content.split('\n')
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            
            print("Context around the error:")
            for i in range(start, end):
                marker = ">>> " if i == e.lineno - 1 else "    "
                print(f"{marker}{i+1:3d}: {lines[i]}")
            
            return False
        
        # Check for common indentation issues
        print("\nChecking for common indentation issues...")
        
        lines = content.split('\n')
        issues_found = False
        
        for i, line in enumerate(lines, 1):
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                print(f"⚠️  Line {i}: Mixed tabs and spaces")
                print(f"    {repr(line)}")
                issues_found = True
            
            # Check for inconsistent indentation
            if line.strip() and not line.startswith('#'):
                indent = len(line) - len(line.lstrip())
                if indent % 4 != 0 and indent > 0:
                    print(f"⚠️  Line {i}: Indentation not multiple of 4")
                    print(f"    Indent: {indent} spaces")
                    print(f"    {line}")
                    issues_found = True
        
        if not issues_found:
            print("✅ No common indentation issues found")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error checking file: {e}")
        return False

def fix_common_issues(filename, output_filename=None):
    """Fix common indentation issues"""
    if output_filename is None:
        output_filename = filename.replace('.py', '_fixed.py')
    
    print(f"\nAttempting to fix common issues...")
    print(f"Output file: {output_filename}")
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        changes_made = 0
        
        for i, line in enumerate(lines, 1):
            original_line = line
            
            # Replace tabs with 4 spaces
            if '\t' in line:
                line = line.expandtabs(4)
                changes_made += 1
                print(f"Fixed tabs on line {i}")
            
            # Fix common indentation patterns
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                
                # Round to nearest multiple of 4
                if indent % 4 != 0 and indent > 0:
                    new_indent = round(indent / 4) * 4
                    line = ' ' * new_indent + stripped
                    changes_made += 1
                    print(f"Fixed indentation on line {i}: {indent} -> {new_indent}")
            
            fixed_lines.append(line)
        
        # Write fixed file
        with open(output_filename, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"\n✅ Fixed {changes_made} issues")
        print(f"✅ Saved to: {output_filename}")
        
        # Test the fixed file
        print("\nTesting fixed file...")
        return check_indentation(output_filename)
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_indentation.py <filename.py> [--fix]")
        print("Example: python3 check_indentation.py test_01.py --fix")
        return
    
    filename = sys.argv[1]
    should_fix = '--fix' in sys.argv
    
    # Check the file
    success = check_indentation(filename)
    
    if not success and should_fix:
        print("\n" + "="*50)
        print("ATTEMPTING TO FIX ISSUES")
        print("="*50)
        fix_common_issues(filename)
    elif not success:
        print("\n💡 Tip: Use --fix flag to attempt automatic fixes")
        print(f"   python3 check_indentation.py {filename} --fix")

if __name__ == "__main__":
    main()
