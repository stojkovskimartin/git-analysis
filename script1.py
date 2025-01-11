import os
import re
from git import Repo, GitCommandError, InvalidGitRepositoryError, BadName
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# Function to clean commit messages by removing non-printable characters
def sanitize_message(message):
    # Remove non-printable characters (ASCII control characters)
    return re.sub(r'[^\x20-\x7E]+', '', message)


# Analyze each repository for commit and conflict information
def analyze_repo(repo_path):
    repo_name = os.path.basename(repo_path)

    try:
        repo = Repo(repo_path)

        # Track merge commit details
        merge_commit_details = []
        total_merges = 0

        # Check only merge commits for conflicts and details
        for commit in repo.iter_commits():
            if len(commit.parents) > 1:  # This is a merge commit
                try:
                    # Collect merge commit details
                    commit_id = commit.hexsha
                    parent_merge_id = commit.parents[0].hexsha if commit.parents else ""
                    author_name = commit.author.name
                    author_email = commit.author.email
                    date_time = commit.committed_datetime.strftime("%a %b %d %H:%M:%S %Y")

                    # Sanitize the commit message to remove illegal characters
                    message = sanitize_message(commit.message.strip())

                    # Extract conflicted files from the message if any
                    conflicted_files = []
                    if "Conflicts:" in message:
                        conflicted_files = message.split("Conflicts:")[1].strip().split("\n")

                    if conflicted_files:
                        total_merges += 1
                        merge_commit_details.append({
                            "Project Name": repo_name,
                            "Commit ID": commit_id,
                            "Parent Merge Id": parent_merge_id,
                            "Author Name": author_name,
                            "Author Email": author_email,
                            "DateTime": date_time,
                            "Message": message,
                            "Conflicted Files": ", ".join(conflicted_files)  # Join conflicted files as a single string
                        })
                except GitCommandError as e:
                    print(f"Error analyzing commit {commit_id} in {repo_name}: {e}")
                    continue
                except BadName as e:
                    print(f"Invalid reference encountered in {repo_name}: {e}")
                    continue

        return merge_commit_details
    except InvalidGitRepositoryError:
        print(f"{repo_name} is not a valid git repository.")
        return []
    except Exception as e:
        print(f"Failed to analyze {repo_name}: {e}")
        return []  # Return an empty list if there's an error


# Main function to create the report
def create_report():
    repo_base_path = r"C:\Users\martinstojkovski\PycharmProjects\pythonProject2\cloned_repositories"
    output_path = r"C:\Users\martinstojkovski\PycharmProjects\pythonProject2\GitHub_Project_Merge_Analysis_Report.xlsx"

    # Collect paths to each repository directory
    repo_paths = [os.path.join(repo_base_path, repo_name) for repo_name in os.listdir(repo_base_path) if
                  os.path.isdir(os.path.join(repo_base_path, repo_name))]

    # Initialize a list to collect results from each thread
    all_results = []

    # Use ThreadPoolExecutor to analyze repositories concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on system resources
        # Submit each repository to the executor
        futures = {executor.submit(analyze_repo, repo_path): repo_path for repo_path in repo_paths}

        # Collect results as each thread completes
        for future in as_completed(futures):
            repo_results = future.result()
            if repo_results:
                all_results.extend(repo_results)  # Append each repository's results to the main list

    # Save all results to Excel if there are any entries
    if all_results:
        # Convert results to DataFrame and save as Excel
        df = pd.DataFrame(all_results)
        df.to_excel(output_path, index=False)
        print(f"Report saved to {output_path}")
    else:
        print("No repositories were successfully analyzed.")


# Run the report creation
if __name__ == "__main__":
    create_report()
