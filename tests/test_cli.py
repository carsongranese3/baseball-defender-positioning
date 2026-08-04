"""Integration tests: drive display.py end to end.

These build their own synthetic dataset in a temp directory, so they pass on a
fresh clone with no downloaded data. Run from the repo root:

    python -m unittest discover -s tests -v
"""
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Statcast pixel space -> feet, inverted from optimizer.transform_coordinates.
def to_pixels(x_ft, y_ft):
    return (x_ft / 2.43) + 125, 204.5 - (y_ft / 2.43)


def build_fixture(root, infield_pts, outfield_pts):
    """Write a minimal but valid Batters/Pitchers tree plus the two scripts."""
    bat = os.path.join(root, "Batters", "test_guy")
    pit = os.path.join(root, "Pitchers", "test_arm")
    os.makedirs(bat, exist_ok=True)
    os.makedirs(pit, exist_ok=True)
    for f in ("display.py", "optimizer.py"):
        shutil.copy(os.path.join(REPO, f), os.path.join(root, f))

    cols = ["hc_x", "hc_y", "pitch_type", "release_speed", "launch_angle", "events"]
    rows = []
    for x, y in infield_pts:
        hx, hy = to_pixels(x, y)
        rows.append([hx, hy, "FF", 95.0, 15.0, "field_out"])
    for x, y in outfield_pts:
        hx, hy = to_pixels(x, y)
        rows.append([hx, hy, "FF", 95.0, 25.0, "single"])
    with open(os.path.join(bat, "2025_data.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

    with open(os.path.join(pit, "pitches.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pitch_type", "pitch_name", "Count", "Avg_Vel", "Max_Vel", "Usage_%"])
        w.writerow(["FF", "4-Seam Fastball", 100, 95.0, 99.0, 100.0])
    return root


DENSE_IF = [(-60, 100), (-30, 90), (-10, 110), (10, 100), (30, 95),
            (50, 105), (-45, 120), (20, 115), (-5, 95), (40, 90)]
DENSE_OF = [(-150, 300), (-60, 320), (0, 340), (60, 310), (150, 290),
            (-100, 280), (100, 330)]


class DisplayRunner(unittest.TestCase):
    """Base class: one shared fixture directory per subclass."""
    INFIELD = DENSE_IF
    OUTFIELD = DENSE_OF

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bdp-test-")
        build_fixture(cls.tmp, cls.INFIELD, cls.OUTFIELD)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_display(self, stdin_text, timeout=120):
        proc = subprocess.run([sys.executable, "display.py"], input=stdin_text,
                              capture_output=True, text=True, cwd=self.tmp,
                              env=dict(os.environ, MPLBACKEND="Agg"),
                              timeout=timeout)
        return proc.stdout + proc.stderr, proc.returncode

    def assert_no_traceback(self, out):
        self.assertNotIn("Traceback", out)
        self.assertNotIn("ValueError", out)
        self.assertNotIn("IndexError", out)


VALID = "0\n0\n1\n13\n8\n"   # only one batter and one pitcher in the fixture


class TestInputValidation(DisplayRunner):
    """Every prompt must re-prompt on bad input instead of raising."""

    def test_non_numeric_batter(self):
        out, _ = self.run_display("abc\n" + VALID)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_out_of_range_batter(self):
        out, _ = self.run_display("99\n" + VALID)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_negative_index_rejected(self):
        """-1 used to silently select the last batter via negative indexing."""
        out, _ = self.run_display("-1\n" + VALID)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_float_index_rejected(self):
        out, _ = self.run_display("1.5\n" + VALID)
        self.assert_no_traceback(out)
        self.assertIn("Enter a number from 0 to", out)

    def test_out_of_range_outs(self):
        out, _ = self.run_display("0\n0\n9\n1\nNone\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter a whole number from 0 to 2", out)

    def test_invalid_inning(self):
        out, _ = self.run_display("0\n0\n1\nNone\n0\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter a whole number from 1 to 30", out)

    def test_garbage_runners(self):
        """A stray digit used to place a phantom runner."""
        out, _ = self.run_display("0\n0\n1\nxyz\nNone\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter None, or any combination", out)

    def test_invalid_base_numbers(self):
        out, _ = self.run_display("0\n0\n1\n456\nNone\n1\n")
        self.assert_no_traceback(out)
        self.assertIn("Enter None, or any combination", out)

    def test_eof_exits_cleanly(self):
        out, code = self.run_display("")
        self.assert_no_traceback(out)
        self.assertIn("Cancelled", out)
        self.assertEqual(code, 0)


class TestSituationalRules(DisplayRunner):
    def test_holding_double_play_and_infield_in(self):
        out, code = self.run_display("0\n0\n1\n13\n8\n")
        self.assertEqual(code, 0)
        self.assertIn("Holding Runner", out)
        self.assertIn("Double Play Depth", out)
        self.assertIn("Infield IN", out)

    def test_no_doubles_late_with_two_outs(self):
        out, _ = self.run_display("0\n0\n2\nNone\n9\n")
        self.assertIn("No Doubles", out)

    def test_no_strategies_early_bases_empty(self):
        out, _ = self.run_display("0\n0\n0\nNone\n1\n")
        self.assertNotIn("STRATEGY", out)

    def test_runner_order_normalized(self):
        """'31' means the same as '13'."""
        out, _ = self.run_display("0\n0\n1\n31\n8\n")
        self.assertIn("Holding Runner", out)
        self.assertIn("Infield IN", out)

    def test_clean_run_no_traceback(self):
        out, code = self.run_display(VALID)
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", out)


class TestSparseDataFallback(DisplayRunner):
    """Too few batted balls to cluster -> hardcoded fallback positions.
    Regression: the situational adjustments used to be skipped entirely here."""
    INFIELD = [(-30, 90), (10, 100), (40, 80)]     # 3 -> infield fallback
    OUTFIELD = [(-150, 300), (60, 310)]            # 2 -> outfield fallback

    def test_fallback_still_runs(self):
        out, code = self.run_display("0\n0\n1\n3\n8\n")
        self.assertEqual(code, 0)
        self.assert_no_traceback(out)

    def test_infield_in_announced_in_fallback(self):
        out, _ = self.run_display("0\n0\n1\n3\n8\n")
        self.assertIn("Infield IN", out)

    def test_no_doubles_announced_in_fallback(self):
        out, _ = self.run_display("0\n0\n2\nNone\n9\n")
        self.assertIn("No Doubles", out)


class TestMissingData(unittest.TestCase):
    def test_empty_repo_gives_message_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "Batters"))
            os.makedirs(os.path.join(tmp, "Pitchers"))
            for f in ("display.py", "optimizer.py"):
                shutil.copy(os.path.join(REPO, f), os.path.join(tmp, f))
            proc = subprocess.run([sys.executable, "display.py"], input="",
                                  capture_output=True, text=True, cwd=tmp,
                                  env=dict(os.environ, MPLBACKEND="Agg"),
                                  timeout=120)
            self.assertNotIn("Traceback", proc.stdout + proc.stderr)
            self.assertIn("Missing data", proc.stdout)

    def test_missing_arsenal_gives_message_not_crash(self):
        """A pitcher folder with season data but no pitches.csv."""
        with tempfile.TemporaryDirectory() as tmp:
            build_fixture(tmp, DENSE_IF, DENSE_OF)
            os.remove(os.path.join(tmp, "Pitchers", "test_arm", "pitches.csv"))
            proc = subprocess.run([sys.executable, "display.py"],
                                  input="0\n0\n1\nNone\n1\n",
                                  capture_output=True, text=True, cwd=tmp,
                                  env=dict(os.environ, MPLBACKEND="Agg"),
                                  timeout=120)
            self.assertNotIn("Traceback", proc.stdout + proc.stderr)
            self.assertIn("No pitches.csv", proc.stdout)


def repo_statcast_files():
    return [f for f in glob.glob(os.path.join(REPO, "Batters", "*", "*.csv"))
            + glob.glob(os.path.join(REPO, "Pitchers", "*", "*.csv"))
            if not f.endswith("pitches.csv")]


@unittest.skipUnless(repo_statcast_files(),
                     "no downloaded data in this checkout (expected on a fresh clone)")
class TestDataContract(unittest.TestCase):
    """Only runs when the user has downloaded data. Checks that what the
    downloaders produce still contains every column the code reads."""

    REQUIRED = ["hc_x", "hc_y", "pitch_type", "release_speed",
                "launch_angle", "pitch_name"]

    def test_required_columns_present(self):
        for f in repo_statcast_files():
            with open(f) as fh:
                header = next(csv.reader(fh))
            for col in self.REQUIRED:
                self.assertIn(col, header, f"{os.path.basename(f)} missing {col}")

    def test_arsenal_files_have_needed_columns(self):
        for f in glob.glob(os.path.join(REPO, "Pitchers", "*", "pitches.csv")):
            with open(f) as fh:
                header = next(csv.reader(fh))
            for col in ("pitch_type", "Usage_%", "Avg_Vel"):
                self.assertIn(col, header, f"{f} missing {col}")

    def test_pop_out_event_absent(self):
        """Justifies removing the `events != 'pop_out'` filter."""
        for f in glob.glob(os.path.join(REPO, "Batters", "*", "*.csv")):
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    self.assertNotEqual(row.get("events"), "pop_out",
                                        f"pop_out reappeared in {f}")


if __name__ == "__main__":
    unittest.main()
