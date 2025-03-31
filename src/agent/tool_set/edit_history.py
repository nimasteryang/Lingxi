"""History management for file edits with disk-based storage and memory constraints."""

import tempfile
from pathlib import Path
from typing import Optional

from diskcache import Cache


class FileHistoryManager:
    """Manages file edit history with disk-based storage and memory constraints."""

    def __init__(
        self, max_history_per_file: int = 5, history_dir: Optional[Path] = None
    ):
        """Initialize the history manager.

        Args:
            max_history_per_file: Maximum number of history entries to keep per file (default: 5)
            history_dir: Directory to store history files. If None, uses a temp directory

        Notes:
            - Each file's history is limited to the last N entries to conserve memory
            - The disk cache is limited to 500MB total to prevent excessive disk usage
            - Older entries are automatically removed when limits are exceeded
        """
        self.max_history_per_file = max_history_per_file
        if history_dir is None:
            history_dir = Path(tempfile.mkdtemp(prefix='oh_editor_history_'))
        self.cache = Cache(str(history_dir), size_limit=5e8)  # 500MB size limit

    def add_history(self, file_path: Path, content: str):
        """Add a new history entry for a file."""
        key = str(file_path)
        # Get list of entry indices and counter for this file
        entries_key = f'{key}:entries'
        counter_key = f'{key}:counter'
        entries = self.cache.get(entries_key, [])
        counter = self.cache.get(counter_key, 0)

        # Add new entry with monotonically increasing counter
        entry_key = f'{key}:{counter}'
        self.cache.set(entry_key, content)
        entries.append(entry_key)
        counter += 1

        # Keep only last N entries
        if len(entries) > self.max_history_per_file:
            old_key = entries.pop(0)
            self.cache.delete(old_key)

        # Update entries list and counter
        self.cache.set(entries_key, entries)
        self.cache.set(counter_key, counter)

    def get_last_history(self, file_path: Path) -> Optional[str]:
        """Get the most recent history entry for a file."""
        key = str(file_path)
        entries_key = f'{key}:entries'
        entries = self.cache.get(entries_key, [])

        if not entries:
            return None

        # Get and remove last entry
        last_key = entries.pop()
        content = self.cache.get(last_key)
        self.cache.delete(last_key)

        # Update entries list
        self.cache.set(entries_key, entries)
        return content

    def clear_history(self, file_path: Path):
        """Clear history for a given file."""
        key = str(file_path)
        entries_key = f'{key}:entries'
        counter_key = f'{key}:counter'
        entries = self.cache.get(entries_key, [])

        # Delete all entries
        for entry_key in entries:
            self.cache.delete(entry_key)

        # Delete entries list and counter
        self.cache.delete(entries_key)
        self.cache.delete(counter_key)