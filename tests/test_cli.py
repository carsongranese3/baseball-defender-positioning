"""Integration tests: drive display.py end to end and check the data contract.

These launch display.py as a subprocess with piped stdin, so they exercise the
real prompt loop. Run from the repo root:

    python -m unittest discover -s tests -v
"""
import csv
import glob
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_display(stdin_text, timeout=180):
    """Run display.py headless with piped input; return (stdout+stderr, code)."""
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "display.py"], input=stdin_text,
                          capture_output=True, text=True, cwd=REPO, env=env,
                          timeout=timeout)
    return proc.stdout + proc.stderr, proc.returncode


# Index 1 is cal_raleigh (fewest seasons) - keeps these tests quick.
VALID_RUN = "1\n1\n1\n13\n8\n"


class TestInputValidation(unittest.TestCase):
    """Every prompt must re-prompt on bad input instead of raising."""

    def assert_no_traceback(self, out):
        self.assertNotIn("Traceback", out)
        self.assertNotIn("ValueError", out)
        self.assertNotIn("IndexError", out)

    def test_non_numeric_batter(self):
        out, _ = run_display("abc\n" + VALID_RUN)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_out_of_range_batter(self):
        out, _ = run_display("99\n" + VALID_RUN)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_negative_index_rejected(self):
        """-1 used to silently select the last batter via negative indexing."""
        out, _ = run_display("-1\n" + VALID_RUN)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_out_of_range_outs(self):
        out, _ = run_display("1\n1\n9\n1\nNone\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter a whole number from 0 to 2", out)

    def test_invalid_inning(self):
        out, _ = run_display("1\n1\n1\nNone\n0\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter a whole number from 1 to 30", out)

    def test_garbage_runners(self):
        """A stray digit used to place a phantom runner."""
        out, _ = run_display("1\n1\n1\nxyz\nNone\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter None, or any combination", out)

    def test_eof_exits_cleanly(self):
        out, code = run_display("")
        self.assert_no_traceback(out)
        self.assertIn("Cancelled", out)
        self.assertEqual(code, 0)


class TestSituationalRules(unittest.TestCase):
    def test_holding_runner_and_double_play_and_infield_in(self):
        out, code = run_display("1\n1\n1\n13\n8\n")
        self.assertEqual(code, 0)
        self.assertIn("Holding Runner", out)
        self.assertIn("Double Play Depth", out)
        self.assertIn("Infield IN", out)

    def test_no_doubles_late_with_two_outs(self):
        out, _ = run_display("1\n1\n2\nNone\n9\n")
        self.assertIn("No Doubles", out)

    def test_no_strategies_early_with_bases_empty(self):
        out, _ = run_display("1\n1\n0\nNone\n1\n")
        self.assertNotIn("STRATEGY", out)

    def test_runner_order_is_normalized(self):
        """'31' means the same as '13'."""
        out, _ = run_display("1\n1\n1\n31\n8\n")
        self.assertIn("Holding Runner", out)
        self.assertIn("Infield IN", out)

    def test_clean_run_has_no_traceback(self):
        out, code = run_display(VALID_RUN)
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", out)


class TestMissingDataMessages(unittest.TestCase):
    def test_no_players_is_handled(self):
        """Empty Batters/Pitchers dirs give a message, not a crash."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "Batters"))
            os.makedirs(os.path.join(tmp, "Pitchers"))
            for f in ("display.py", "optimizer.py"):
                with open(os.path.join(REPO, f)) as src, \
                     open(os.path.join(tmp, f), "w") as dst:
                    dst.write(src.read())
            proc = subprocess.run([sys.executable, "display.py"], input="",
                                  capture_output=True, text=True, cwd=tmp,
                                  env=dict(os.environ, MPLBACKEND="Agg"),
                                  timeout=120)
            self.assertNotIn("Traceback", proc.stdout + proc.stderr)
            self.assertIn("Missing data", proc.stdout)


class TestDataContract(unittest.TestCase):
    """The columns the code reads must exist in every Statcast CSV."""

    REQUIRED = ["hc_x", "hc_y", "pitch_type", "release_speed",
                "launch_angle", "pitch_name"]

    def statcast_files(self):
        return [f for f in glob.glob(os.path.join(REPO, "Batters", "*", "*.csv"))
                + glob.glob(os.path.join(REPO, "Pitchers", "*", "*.csv"))
                if not f.endswith("pitches.csv")]

    def test_required_columns_present(self):
        files = self.statcast_files()
        self.assertGreater(len(files), 0, "no Statcast CSVs found")
        for f in files:
            with open(f) as fh:
                header = next(csv.reader(fh))
            for col in self.REQUIRED:
                self.assertIn(col, header, f"{os.path.basename(f)} missing {col}")

    def test_arsenal_files_have_needed_columns(self):
        files = glob.glob(os.path.join(REPO, "Pitchers", "*", "pitches.csv"))
        self.assertGreater(len(files), 0, "no pitches.csv found")
        for f in files:
            with open(f) as fh:
                header = next(csv.reader(fh))
            for col in ("pitch_type", "Usage_%", "Avg_Vel"):
                self.assertIn(col, header, f"{f} missing {col}")

    def test_pop_out_event_really_is_absent(self):
        """Justifies removing the `events != 'pop_out'` filter. If a future
        download reintroduces it, popups would silently rejoin the infield set."""
        for f in glob.glob(os.path.join(REPO, "Batters", "*", "*.csv")):
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    self.assertNotEqual(row.get("events"), "pop_out",
                                        f"pop_out reappeared in {f}")


if __name__ == "__main__":
    unittest.main()
