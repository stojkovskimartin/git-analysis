import os
import subprocess
import time
import xml.etree.ElementTree as ET

import pandas as pd
from git import Repo, GitCommandError


class MergeAnalyzer:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.results = []

    def remove_git_locks(self):
        """Remove Git lock files if they exist."""
        lock_files = [
            os.path.join(self.repo_path, '.git', 'index.lock'),
            os.path.join(self.repo_path, '.git', 'HEAD.lock'),
            os.path.join(self.repo_path, '.git', 'refs', 'heads', '*.lock'),
            os.path.join(self.repo_path, '.git', 'MERGE_HEAD.lock')
        ]
        for lock_file in lock_files:
            try:
                if '*' in lock_file:
                    lock_dir = os.path.dirname(lock_file)
                    if os.path.exists(lock_dir):
                        for file in os.listdir(lock_dir):
                            if file.endswith('.lock'):
                                full_path = os.path.join(lock_dir, file)
                                if os.path.exists(full_path):
                                    os.remove(full_path)
                else:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
            except Exception as e:
                print(f"Warning: Could not remove lock file {lock_file}: {str(e)}")

    def get_merge_parents(self, merge_commit):
        """Get both parents of a merge commit."""
        if len(merge_commit.parents) != 2:
            raise ValueError(f"Commit {merge_commit.hexsha} is not a merge commit")
        return merge_commit.parents[0], merge_commit.parents[1]

    def reset_to_clean_state(self):
        """Reset repository to a clean state."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.remove_git_locks()
                try:
                    self.repo.git.merge('--abort')
                except GitCommandError:
                    pass
                self.repo.git.reset('--hard')
                self.repo.git.clean('-fd')
                return True
            except GitCommandError as e:
                print(f"Warning: Error during cleanup (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
        print("Failed to reset repository after multiple attempts")
        return False

    def try_automatic_merge(self, base, branch, strategy):
        """Attempt automatic merge with specified strategy."""
        merge_success = False
        error_message = None

        if not self.reset_to_clean_state():
            return False, "Failed to reset repository to clean state"

        try:
            print(f"Checking out base commit: {base.hexsha}")
            self.repo.git.checkout(base.hexsha)
            try:
                print(f"Attempting merge with strategy: {strategy}")
                if strategy == 'ours':
                    self.repo.git.merge(branch.hexsha, strategy='ours', no_commit=True)
                else:
                    self.repo.git.merge(branch.hexsha, strategy=strategy, no_commit=True)
                merge_success = True
                print("Merge successful.")
            except GitCommandError as e:
                error_message = str(e)
                print(f"Merge failed: {error_message}")
                # Print the diff if there's a conflict
                self.print_diff(base, branch)  # Print differences on failure
                merge_success = False
        except Exception as e:
            error_message = str(e)
            print(f"Exception during merge: {error_message}")
            merge_success = False
        finally:
            self.reset_to_clean_state()

        return merge_success, error_message

    def print_diff(self, base_commit, target_commit):
        """Print a summary of the differences between two commits."""
        try:
            # Get the diff output
            diff_output = self.repo.git.diff(base_commit.hexsha, target_commit.hexsha)

            if diff_output:
                # If there are differences, print a summary
                print(f"\nDifferences detected between {base_commit.hexsha} and {target_commit.hexsha}:")
                # Count lines added and removed for summary
                lines_added = sum(
                    1 for line in diff_output.splitlines() if line.startswith('+') and not line.startswith('+++'))
                lines_removed = sum(
                    1 for line in diff_output.splitlines() if line.startswith('-') and not line.startswith('---'))
                print(f"  Lines added: {lines_added}, Lines removed: {lines_removed}")
            else:
                print("No differences found.")
        except Exception as e:
            print(f"Error printing diff: {str(e)}")

    def calculate_diff(self, commit1, commit2):
        """Calculate the difference between two commits."""
        print(f"Calculating diff between {commit1.hexsha} and {commit2.hexsha}")
        try:
            diff_stats = self.repo.git.diff(commit1.hexsha, commit2.hexsha, numstat=True)
            total_lines_added = 0
            total_lines_removed = 0
            for line in diff_stats.splitlines():
                if line.strip():
                    added, removed, _ = line.split('\t')
                    if added != '-' and removed != '-':
                        total_lines_added += int(added)
                        total_lines_removed += int(removed)
            total_lines_changed = total_lines_added + total_lines_removed
            print(
                f"Diff result: {total_lines_changed} lines changed, {total_lines_added} added, {total_lines_removed} removed")
            return {
                'total_lines_changed': total_lines_changed,
                'lines_added': total_lines_added,
                'lines_removed': total_lines_removed
            }
        except GitCommandError as e:
            print(f"Error calculating diff: {str(e)}")
            return {'total_lines_changed': 0, 'lines_added': 0, 'lines_removed': 0}

    def analyze_merge_commit(self, merge_commit_hash):
        """Analyze a specific merge commit and its automatic merge alternatives."""
        if not self.reset_to_clean_state():
            print(f"Skipping {merge_commit_hash}: Could not reset to clean state")
            return None

        try:
            merge_commit = self.repo.commit(merge_commit_hash)
            if len(merge_commit.parents) != 2:
                print(f"Skipping {merge_commit_hash}: Not a merge commit")
                return None

            parent1, parent2 = self.get_merge_parents(merge_commit)
            committed_date = merge_commit.committed_datetime.replace(tzinfo=None)

            results = {
                'commit_hash': merge_commit.hexsha,
                'author': merge_commit.author.name,
                'date': committed_date,
                'original_message': merge_commit.message,
            }

            # Analyze original merge
            print(f"\nAnalyzing original merge: {merge_commit.hexsha}")
            self.repo.git.checkout(merge_commit.hexsha)
            original_compilable, original_compile_output = self.compile_project()
            original_tests_run, original_total_tests, original_failed_tests, original_test_output = self.run_tests()

            results['original_compilable'] = original_compilable
            results['original_tests_run'] = original_tests_run
            results['original_total_tests'] = original_total_tests
            results['original_failed_tests'] = original_failed_tests

            # Calculate diff for original merge
            original_diff = self.calculate_diff(parent1, merge_commit)
            results['original_diff'] = original_diff  # Store diff in results
            print(f"Original merge diff: {original_diff}")

            strategies = ['recursive', 'resolve', 'octopus', 'ours']

            for strategy in strategies:
                print(f"\nAttempting merge with strategy: {strategy}")
                success, error = self.try_automatic_merge(parent1, parent2, strategy)
                results[f'{strategy}_success'] = success

                if success:
                    head_commit = self.repo.head.commit
                    print(f"Merge successful. Head commit: {head_commit.hexsha}")

                    compilable, compile_output = self.compile_project()
                    tests_run, total_tests, failed_tests, test_output = self.run_tests()

                    # Calculate diff and information loss
                    strategy_diff = self.calculate_diff(parent1, head_commit)
                    info_loss = self.calculate_information_loss(merge_commit, head_commit)

                    # Store additional information in results
                    results[f'{strategy}_compilable'] = compilable
                    results[f'{strategy}_tests_run'] = tests_run
                    results[f'{strategy}_total_tests'] = total_tests
                    results[f'{strategy}_failed_tests'] = failed_tests
                    results[f'{strategy}_diff'] = strategy_diff  # Store strategy diff
                    results[f'{strategy}_info_loss'] = info_loss['information_loss_percentage']
                else:
                    print(f"Merge failed: {error}")
                    results[f'{strategy}_error'] = error

            results['auto_merge_possible'] = any(results[f'{s}_success'] for s in strategies)

            return results  # Ensure this returns all relevant data
        except Exception as e:
            print(f"Error analyzing merge commit {merge_commit_hash}: {str(e)}")
            return None
        finally:
            self.reset_to_clean_state()

    def calculate_information_loss(self, original_commit, new_commit):
        """Calculate information loss between original and new merge."""
        try:
            print(f"Calculating information loss between:")
            print(f"Original commit: {original_commit.hexsha}")
            print(f"New commit: {new_commit.hexsha}")

            diff = self.repo.git.diff(original_commit.hexsha, new_commit.hexsha)
            added_lines = sum(1 for line in diff.splitlines() if line.startswith('+'))
            removed_lines = sum(1 for line in diff.splitlines() if line.startswith('-'))
            total_lines = len(diff.splitlines())

            print(f"Diff stats: Added lines: {added_lines}, Removed lines: {removed_lines}, Total lines: {total_lines}")

            if total_lines == 0:
                print("No differences found, information loss is 0%")
                return {'information_loss_percentage': 0}

            info_loss_percentage = (removed_lines / total_lines) * 100
            print(f"Calculated information loss: {info_loss_percentage}%")

            return {
                'added_lines': added_lines,
                'removed_lines': removed_lines,
                'total_diff_lines': total_lines,
                'information_loss_percentage': info_loss_percentage
            }
        except GitCommandError as e:
            print(f"Error in calculate_information_loss: {str(e)}")
            return {'error': str(e)}

    def compile_project(self):
        """Compile the project using Maven."""
        try:
            result = subprocess.run(['mvn', 'compile'], cwd=self.repo_path, capture_output=True, text=True, timeout=300)
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out after 5 minutes"
        except Exception as e:
            return False, str(e)

    def run_tests(self):
        """Run JUNIT tests using Maven."""
        try:
            result = subprocess.run(['mvn', 'test'], cwd=self.repo_path, capture_output=True, text=True, timeout=600)
            return self.parse_test_results(result.stdout)
        except subprocess.TimeoutExpired:
            return False, 0, 0, "Tests timed out after 10 minutes"
        except Exception as e:
            return False, 0, 0, str(e)

    def parse_test_results(self, output):
        """Parse Maven test output to get test results."""
        try:
            surefire_reports = os.path.join(self.repo_path, 'target', 'surefire-reports')
            if not os.path.exists(surefire_reports):
                return False, 0, 0, "No test reports found"

            total_tests = 0
            failures = 0
            for file in os.listdir(surefire_reports):
                if file.endswith('.xml'):
                    tree = ET.parse(os.path.join(surefire_reports, file))
                    root = tree.getroot()
                    total_tests += int(root.attrib.get('tests', 0))
                    failures += int(root.attrib.get('failures', 0)) + int(root.attrib.get('errors', 0))

            return True, total_tests, failures, f"Total: {total_tests}, Failed: {failures}"
        except Exception as e:
            return False, 0, 0, f"Error parsing test results: {str(e)}"

    def calculate_information_loss(self, original_commit, new_commit):
        """Calculate information loss between original and new merge."""
        try:
            diff = self.repo.git.diff(original_commit.hexsha, new_commit.hexsha)
            added_lines = sum(1 for line in diff.splitlines() if line.startswith('+'))
            removed_lines = sum(1 for line in diff.splitlines() if line.startswith('-'))
            total_lines = len(diff.splitlines())
            return {
                'added_lines': added_lines,
                'removed_lines': removed_lines,
                'total_diff_lines': total_lines,
                'information_loss_percentage': (removed_lines / total_lines) * 100 if total_lines > 0 else 0
            }
        except GitCommandError as e:
            return {'error': str(e)}


def analyze_repository(repo_path, commit_list):
    """Analyze a repository for specific merge commits."""
    analyzer = MergeAnalyzer(repo_path)
    results_list = []
    analyzer.remove_git_locks()

    total_commits = len(commit_list)
    for i, commit_hash in enumerate(commit_list, 1):
        print(f"\nAnalyzing commit {i}/{total_commits}: {commit_hash}")
        result = analyzer.analyze_merge_commit(commit_hash)
        if result:
            results_list.append(result)

    # Create a DataFrame from the list of result dictionaries
    results_df = pd.DataFrame(results_list)

    return results_df  # Return complete DataFrame with all data captured


def main():
    # Configuration
    repo_path = r"C:\Users\martinstojkovski\PycharmProjects\pythonProject2\cloned_repositories\neo4j-framework"
    excel_path = "commits.xlsx"
    output_path = "merge_analysis_results.xlsx"

    # Read commits from Excel
    commits_df = pd.read_excel(excel_path)
    commit_list = commits_df['commit_hash'].tolist()

    # Analyze repository
    results_df = analyze_repository(repo_path, commit_list)

    # Save results
    if not results_df.empty:
        if 'date' in results_df.columns:
            results_df['date'] = results_df['date'].apply(lambda x: x.replace(tzinfo=None) if x is not None else x)
        results_df.to_excel(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Print comprehensive summary
        print("\nAnalysis Summary:")
        print(f"Total commits analyzed: {len(results_df)}")
        print(f"Commits that could be auto-merged: {results_df['auto_merge_possible'].sum()}")

        # Detailed statistics for successful merges
        successful_merges = results_df[results_df['auto_merge_possible']]
        if not successful_merges.empty:
            print("\nFor commits that could be auto-merged:")
            strategies = ['recursive', 'resolve', 'octopus', 'ours']
            for strategy in strategies:
                strategy_success = successful_merges[f'{strategy}_success'].sum()
                print(f"\n'{strategy}' strategy:")
                print(f"  Successful merges: {strategy_success}")
                if strategy_success > 0:
                    compilable = successful_merges[f'{strategy}_compilable'].sum()
                    print(f"  Compilable: {compilable} ({compilable / strategy_success * 100:.2f}%)")

                    tests_run = successful_merges[f'{strategy}_tests_run'].sum()
                    total_tests = successful_merges[f'{strategy}_total_tests'].sum()
                    failed_tests = successful_merges[f'{strategy}_failed_tests'].sum()
                    if tests_run > 0:
                        print(f"  Tests run: {tests_run} merges")
                        print(f"  Total tests: {total_tests}")
                        print(f"  Failed tests: {failed_tests} ({failed_tests / total_tests * 100:.2f}% of all tests)")

                    avg_info_loss = successful_merges[f'{strategy}_info_loss'].mean()
                    print(f"  Average information loss: {avg_info_loss:.2f}%")

        # Compare with original merges
        print("\nComparison with original merges:")
        original_compilable = results_df['original_compilable'].sum()
        print(f"Original merges compilable: {original_compilable} ({original_compilable / len(results_df) * 100:.2f}%)")

        original_tests_run = results_df['original_tests_run'].sum()
        original_total_tests = results_df['original_total_tests'].sum()
        original_failed_tests = results_df['original_failed_tests'].sum()
        if original_tests_run > 0:
            print(f"Original merges with tests run: {original_tests_run}")
            print(f"Original total tests: {original_total_tests}")
            print(
                f"Original failed tests: {original_failed_tests} ({original_failed_tests / original_total_tests * 100:.2f}% of all tests)")

    else:
        print("\nNo results were generated.")


if __name__ == "__main__":
    main()
