"""
Download and clean Project Gutenberg texts.

Downloads books from Project Gutenberg, strips headers/footers,
and saves cleaned text files (pg*.txt) in the same directory.
"""

import os
import requests

# Get directory of this script
script_dir = os.path.dirname(__file__)

urls = [
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "https://www.gutenberg.org/cache/epub/67979/pg67979.txt",
    "https://www.gutenberg.org/cache/epub/16389/pg16389.txt",
]

for url in urls:
    response = requests.get(url)
    response.raise_for_status()

    # Extract filename from URL and create output path
    filename = url.split('/')[-1]
    output_path = os.path.join(script_dir, filename)

    buf = ''
    lines_in_buf = 0
    started = False

    with open(output_path, 'w') as f:
        for line in response.text.splitlines():
            line = line.strip()
            # Skip until start marker
            if not started:
                if line.startswith('*** START OF THE PROJECT'):
                    started = True
                continue
            # Stop at end marker
            if line.startswith('*** END OF THE PROJECT'):
                break
            # Empty line: flush buffer if it has enough content
            if len(line) == 0:
                if lines_in_buf > 2:
                    f.write(buf + '\n')
                buf = ''
                lines_in_buf = 0
                continue
            # Accumulate line into buffer
            buf += line + ' '
            lines_in_buf += 1

    print(f"Written to {output_path}")
