import csv
import os
import requests
from git import Repo
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# GitHub CSV URL
csv_url = "https://raw.githubusercontent.com/spgroup/s3m/master/svj/replication/projects.csv"
# Directory for cloned repositories
cloned_repos_dir = "cloned_repositories"


# Fetch CSV data from URL
def fetch_csv_data(url):
    response = requests.get(url)
    response.raise_for_status()
    decoded_content = response.content.decode('utf-8')
    return list(csv.reader(decoded_content.splitlines(), delimiter=','))


# Clone repository and get commit and conflict information
def clone_and_analyze_repo(repo_url):
    # Ensure the repo URL is complete
    if not repo_url.startswith("https://github.com/"):
        repo_url = f"https://github.com/{repo_url}"

    repo_name = repo_url.split('/')[-1]
    local_path = os.path.join(cloned_repos_dir, repo_name)

    # Skip cloning if the repository already exists
    if os.path.exists(local_path):
        print(f"Repository {repo_name} already exists. Skipping clone.")
    else:
        try:
            Repo.clone_from(repo_url, local_path)
            print(f"Cloned {repo_url} successfully.")
        except Exception as e:
            print(f"Failed to clone {repo_url}: {e}")
            return None

    # Initialize Repo object
    repo = Repo(local_path)

    # Count commits
    commit_count = sum(1 for _ in repo.iter_commits())

    # Check for conflicts (simplified example)
    conflict_count = len(repo.index.unmerged_blobs())

    return {"Name": repo_name, "URL": repo_url, "Commit Count": commit_count, "Conflict Count": conflict_count}


# Main function to create the report
def create_report():
    # Ensure the base directory for cloned repositories exists
    os.makedirs(cloned_repos_dir, exist_ok=True)

    # Fetch data from CSV
    csv_data = fetch_csv_data(csv_url)

    # Prepare list of repository URLs
    repo_urls = [row[0] for row in csv_data]

    # Initialize results list
    results = []

    # Use ThreadPoolExecutor for concurrent cloning
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_repo = {executor.submit(clone_and_analyze_repo, url): url for url in repo_urls}
        for future in future_to_repo:
            result = future.result()
            if result:
                results.append(result)

    # Save results to Excel if there are any successful entries
    if results:
        df = pd.DataFrame(results)
        df.to_excel("GitHub_Project_Report.xlsx", index=False)
        print("Report saved to GitHub_Project_Report.xlsx")
    else:
        print("No repositories were successfully processed.")


# Run the report creation
if __name__ == "__main__":
    create_report()
