"""
Utility functions for interacting with the GitHub API.

This module provides helper methods to collect and process
git-related information such as issue reports, repository details,
and other GitHub-specific data retrieval operations.

Several Functions in this module require proper GitHub API authentication
and some may depend on the external library`requests`.
"""

import logging
import os
import re

import dotenv
import requests

from agent.constant import REQUEST_TIMEOUT

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

assert (
    "GITHUB_TOKEN" in os.environ
), "Please put your GITHUB_TOKEN in .env at project root!"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Authentication header
headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}


def parse_github_issue_url(issue_url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)"

    match = re.match(pattern, issue_url)

    if match:
        owner = match.group(1)
        project = match.group(2)
        issue_number = match.group(3)
        return owner, project, issue_number
    else:
        # Return None if the URL doesn't match the expected format
        return None, None, None


def get_issue_description(owner, project, issue):
    """Retrieve the issue description
    Args:
        owner (str): Owner of the project
        project (str): Name of the project
        issue (Union[str, int]): Issue ID
    Returns:
        issue_description (str): The corresponding issue description."""
    issue_api_url = f"https://api.github.com/repos/{owner}/{project}/issues/{issue}"

    response = requests.get(issue_api_url, headers=headers, timeout=REQUEST_TIMEOUT)

    # Check for successful response (HTTP status 200)
    if response.status_code == 200:
        issue_data = response.json()
        issue_description = issue_data.get("body", "No description available.")
        return issue_description
    else:
        # Handle errors
        print(f"Error fetching issue details: {response.status_code}")
        print("Response content:", response.text)
        return None


# Fetch issue events to find the one that closed the issue
def get_issue_events(url_):
    response = requests.get(url_, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()


# Fetch issue details to check for a linked PR
def get_issue_details(url_):
    response = requests.get(url_, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()


# Main logic
def get_issue_close_commit(owner, project, issue):
    """Retrieves the commit that closed the pull request corresponding to a given issue.
    Args:
        owner (str): Owner of the project
        project (str): Name of the project
        issue (Union[str, int]): Issue ID
    Returns:
        commit_id_to_return (str): The corresponding commit SHA."""
    # Fetch the issue details
    issue_url = f"https://api.github.com/repos/{owner}/{project}/issues/{issue}"
    event_url = f"https://api.github.com/repos/{owner}/{project}/issues/{issue}/events"
    issue = get_issue_details(issue_url)

    # Check if the issue has a linked pull request
    if "pull_request" in issue:
        pr_url = issue["pull_request"]["url"]
        logger.info(f"Pull Request that closed the issue: {pr_url}")

        # Fetch the pull request details
        pr_response = requests.get(pr_url, headers=headers, timeout=REQUEST_TIMEOUT)
        pr_response.raise_for_status()
        pr_details = pr_response.json()

        # Check for commit associated with the pull request
        if pr_details["merged_at"]:
            logger.info(f"Pull Request was merged at: {pr_details['merged_at']}")
            logger.info(
                f"Commit that closed the issue: {pr_details['merge_commit_sha']}"
            )
        else:
            logger.info("The pull request is not merged yet.")
    else:
        logger.info("No pull request linked to the issue.")
    # Fetch the events of the issue
    events = get_issue_events(event_url)
    commit_id_to_return = ""
    for event in events:
        # print(event)
        if event["event"] == "closed":
            if "commit_id" in event:
                logger.info(f"Commit that closed the issue: {event['commit_id']}")
                commit_id_to_return = event["commit_id"]
            elif "pull_request" in event:
                pr_url = event["pull_request"]["url"]
                logger.info(f"Pull Request that closed the issue: {pr_url}")
    return commit_id_to_return


if __name__ == "__main__":
    # Run the function to get the closing commit or PR
    get_issue_close_commit("tpope", "vim-sensible", "161")
