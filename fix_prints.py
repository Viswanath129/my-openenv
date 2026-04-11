import sys
import re

with open('inference.py', 'r') as f:
    content = f.read()

content = re.sub(r'import os\n', 'import os\nimport sys\n', content, count=1)
content = content.replace('print(', 'print(file=sys.stderr, ')
# Restore the specific prints that need to go to stdout
content = content.replace('print(file=sys.stderr, f"[START]', 'print(f"[START]')
content = content.replace('print(\n        file=sys.stderr, f"[STEP]', 'print(\n        f"[STEP]')
content = content.replace('print(\n        file=sys.stderr, f"[END]', 'print(\n        f"[END]')

with open('inference.py', 'w') as f:
    f.write(content)
