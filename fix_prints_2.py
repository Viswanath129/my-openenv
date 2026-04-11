import re

with open('inference.py', 'r') as f:
    text = f.read()

# Fix the syntax error: print(file=sys.stderr, "msg", ...) -> print("msg", ..., file=sys.stderr)
# We can just change all print(file=sys.stderr, to print( at the start, and add file=sys.stderr to the end of the arguments.
# Actually, since flush=True is almost everywhere, we can just replace flush=True with file=sys.stderr, flush=True
# Let's revert the bad replace first
text = text.replace('print(file=sys.stderr, ', 'print(')

# Now find all prints that should be stderr and append file=sys.stderr
lines = text.split('\n')
for i, line in enumerate(lines):
    if 'print(' in line and not line.strip().startswith('def '):
        # don't touch the official log prints
        if 'f"[START]' in line or 'f"[STEP]' in line or 'f"[END]' in line:
            continue
        
        # Replace flush=True) with file=sys.stderr, flush=True)
        if 'flush=True)' in line:
            lines[i] = line.replace('flush=True)', 'file=sys.stderr, flush=True)')
        elif ')' in line and 'sys.stderr' not in line:
            # find last parenthesis
            last_paren = line.rfind(')')
            lines[i] = line[:last_paren] + ', file=sys.stderr' + line[last_paren:]

with open('inference.py', 'w') as f:
    f.write('\n'.join(lines))
