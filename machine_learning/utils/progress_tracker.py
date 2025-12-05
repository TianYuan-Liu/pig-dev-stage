"""
Progress tracking and time estimation for ML pipeline execution.
Provides real-time progress updates, ETA calculation, and performance metrics.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import threading

try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    )
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class PipelineProgressTracker:
    """
    Comprehensive progress tracker for ML pipeline execution.
    Tracks overall progress, per-tissue phases, and provides ETA estimates.
    """

    PIPELINE_PHASES = [
        "Data Loading",
        "Stage Selection",
        "Preprocessing",
        "Feature Selection",
        "Model Training",
        "Evaluation",
        "Saving Results"
    ]

    def __init__(
        self,
        total_tissues: int,
        include_cross_validation: bool = True,
        history_file: Optional[Path] = None,
        use_rich: bool = True
    ):
        """
        Initialize the progress tracker.

        Args:
            total_tissues: Total number of tissues to process
            include_cross_validation: Whether cross-validation will be performed
            history_file: Path to timing history file for better ETAs
            use_rich: Whether to use rich library for enhanced display
        """
        self.total_tissues = total_tissues
        self.include_cross_validation = include_cross_validation
        self.use_rich = use_rich and RICH_AVAILABLE

        # Calculate total tasks
        self.total_tasks = total_tissues
        if include_cross_validation:
            self.total_tasks += 1  # Add cross-validation as one task

        # Timing tracking
        self.start_time = None
        self.tissue_times = {}
        self.phase_times = {}
        self.current_tissue = None
        self.current_phase = None
        self.completed_tissues = []
        self.failed_tissues = []

        # History for ETA calculation
        self.history_file = history_file or Path("machine_learning/timing_history.json")
        self.timing_history = self._load_timing_history()

        # Moving average for ETA
        self.recent_tissue_times = deque(maxlen=5)

        # Progress bars
        self.main_progress = None
        self.tissue_progress = None
        self.console = Console() if self.use_rich else None

        # Thread safety
        self.lock = threading.Lock()

        # Status tracking
        self.tissue_status = {}
        self.memory_usage = 0

    def _load_timing_history(self) -> Dict:
        """Load historical timing data for better ETA estimates."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Could not load timing history: {e}")
        return {"tissues": {}, "phases": {}, "cross_validation": None}

    def _save_timing_history(self):
        """Save timing data for future runs."""
        try:
            # Update history with current run data
            for tissue, time_taken in self.tissue_times.items():
                if tissue not in self.timing_history["tissues"]:
                    self.timing_history["tissues"][tissue] = []
                self.timing_history["tissues"][tissue].append(time_taken)
                # Keep only last 10 runs
                self.timing_history["tissues"][tissue] = self.timing_history["tissues"][tissue][-10:]

            # Save phase timings
            for phase_key, time_taken in self.phase_times.items():
                if phase_key not in self.timing_history["phases"]:
                    self.timing_history["phases"][phase_key] = []
                self.timing_history["phases"][phase_key].append(time_taken)
                self.timing_history["phases"][phase_key] = self.timing_history["phases"][phase_key][-10:]

            # Create directory if needed
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.history_file, 'w') as f:
                json.dump(self.timing_history, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save timing history: {e}")

    def start_pipeline(self):
        """Start tracking the overall pipeline."""
        self.start_time = time.time()

        if self.use_rich:
            self._start_rich_display()
        else:
            print(f"\n{'='*80}")
            print(f"Starting ML Pipeline: {self.total_tasks} tasks")
            print(f"{'='*80}")

    def _start_rich_display(self):
        """Initialize rich progress display."""
        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Pipeline Progress"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

        self.tissue_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def start_tissue(self, tissue_name: str):
        """Start tracking a tissue processing."""
        with self.lock:
            self.current_tissue = tissue_name
            self.current_phase = None
            tissue_start = time.time()
            self.tissue_times[tissue_name] = {'start': tissue_start, 'phases': {}}
            self.tissue_status[tissue_name] = 'in_progress'

            if self.use_rich:
                self.console.print(f"\n[bold cyan]Starting tissue: {tissue_name}[/bold cyan]")
            else:
                print(f"\nProcessing tissue: {tissue_name}")
                print(f"Progress: [{len(self.completed_tissues)}/{self.total_tissues}]")

    def start_phase(self, phase_name: str):
        """Start tracking a phase within tissue processing."""
        with self.lock:
            if self.current_tissue and phase_name in self.PIPELINE_PHASES:
                self.current_phase = phase_name
                phase_key = f"{self.current_tissue}_{phase_name}"
                self.phase_times[phase_key] = time.time()

                phase_idx = self.PIPELINE_PHASES.index(phase_name)
                progress_pct = (phase_idx / len(self.PIPELINE_PHASES)) * 100

                if not self.use_rich:
                    print(f"  → Phase {phase_idx+1}/{len(self.PIPELINE_PHASES)}: {phase_name}")

    def end_phase(self, phase_name: str):
        """Complete tracking a phase."""
        with self.lock:
            if self.current_tissue and phase_name in self.PIPELINE_PHASES:
                phase_key = f"{self.current_tissue}_{phase_name}"
                if phase_key in self.phase_times:
                    elapsed = time.time() - self.phase_times[phase_key]
                    self.phase_times[phase_key] = elapsed

                    if not self.use_rich:
                        print(f"    ✓ {phase_name} completed in {elapsed:.1f}s")

    def complete_tissue(self, tissue_name: str, success: bool = True):
        """Mark a tissue as completed."""
        with self.lock:
            if tissue_name in self.tissue_times:
                elapsed = time.time() - self.tissue_times[tissue_name]['start']
                self.tissue_times[tissue_name]['elapsed'] = elapsed

                if success:
                    self.completed_tissues.append(tissue_name)
                    self.tissue_status[tissue_name] = 'completed'
                    self.recent_tissue_times.append(elapsed)

                    if not self.use_rich:
                        print(f"✓ {tissue_name} completed in {elapsed:.1f}s")
                else:
                    self.failed_tissues.append(tissue_name)
                    self.tissue_status[tissue_name] = 'failed'

                    if not self.use_rich:
                        print(f"✗ {tissue_name} failed after {elapsed:.1f}s")

                # Update ETA
                self._update_eta()

    def _update_eta(self):
        """Calculate and update ETA based on completed tasks."""
        if not self.recent_tissue_times:
            return

        avg_time = np.mean(self.recent_tissue_times)
        remaining_tissues = self.total_tissues - len(self.completed_tissues) - len(self.failed_tissues)

        eta_seconds = avg_time * remaining_tissues
        if self.include_cross_validation and remaining_tissues == 0:
            # Estimate cross-validation time as 50% of total tissue time
            eta_seconds = sum(self.recent_tissue_times) * 0.5

        if not self.use_rich and eta_seconds > 0:
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            completed_pct = (len(self.completed_tissues) / self.total_tissues) * 100
            print(f"\nProgress: {completed_pct:.1f}% | ETA: {eta_str}")

    def start_cross_validation(self):
        """Start tracking cross-tissue validation."""
        with self.lock:
            self.current_tissue = None
            self.current_phase = "Cross-Tissue Validation"

            if self.use_rich:
                self.console.print("\n[bold magenta]Starting Cross-Tissue Validation[/bold magenta]")
            else:
                print(f"\n{'='*60}")
                print("Starting Cross-Tissue Validation")
                print(f"{'='*60}")

    def update_memory(self, memory_mb: float):
        """Update current memory usage."""
        self.memory_usage = memory_mb

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        elapsed_total = time.time() - self.start_time if self.start_time else 0

        return {
            'total_tissues': self.total_tissues,
            'completed': len(self.completed_tissues),
            'failed': len(self.failed_tissues),
            'in_progress': self.current_tissue,
            'current_phase': self.current_phase,
            'elapsed_time': elapsed_total,
            'memory_usage_mb': self.memory_usage,
            'tissue_times': {
                t: self.tissue_times.get(t, {}).get('elapsed', 0)
                for t in self.completed_tissues
            }
        }

    def finish_pipeline(self):
        """Complete pipeline tracking and save history."""
        elapsed_total = time.time() - self.start_time if self.start_time else 0

        # Save timing history for future runs
        self._save_timing_history()

        if self.use_rich and self.main_progress:
            self.main_progress.stop()

        # Print summary
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {str(timedelta(seconds=int(elapsed_total)))}")
        print(f"Tissues processed: {len(self.completed_tissues)}/{self.total_tissues}")
        if self.failed_tissues:
            print(f"Failed tissues: {', '.join(self.failed_tissues)}")

        # Print timing breakdown
        if self.completed_tissues:
            print("\nTissue Processing Times:")
            for tissue in self.completed_tissues:
                if tissue in self.tissue_times:
                    tissue_time = self.tissue_times[tissue].get('elapsed', 0)
                    print(f"  • {tissue}: {tissue_time:.1f}s")

        return self.get_progress_summary()

    def create_progress_bar(self, total: int, desc: str = "", unit: str = "it") -> tqdm:
        """Create a tqdm progress bar for sub-tasks."""
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            position=1 if self.current_tissue else 0,
            leave=False,
            ncols=80
        )


class PhaseProgressContext:
    """Context manager for tracking pipeline phases."""

    def __init__(self, tracker: PipelineProgressTracker, phase_name: str):
        self.tracker = tracker
        self.phase_name = phase_name

    def __enter__(self):
        self.tracker.start_phase(self.phase_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_phase(self.phase_name)
        return False