---
title: "Pull Request Resources and Template"
---

*Pull requests* provide the mechanism for updating code in your software repository. Project collaborators can open PRs for the repo's administrators to manage and merge. A PR&mdash;so named because you are *requesting* that your changes be *pulled* into the repo&mdash;creates a comparison of your changes against the project's default branch, with additions highlighted in green and deletions in red, and alerts the administrators to any conflicts with the compared branches. Issues and PRs are independently numbered in GitHub as they are created and can be linked to each other with hashtagged notation (see examples below). You may want to prescribe a template for your PRs or provide guidelines in your repo's [contributing guidelines](https://github.com/LLNL/.github/tree/master/community-health/CONTRIBUTING.md). GitHub provides [extensive information](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests/) on PRs including details on managing branches and reviewing PRs. The PR process can also include automated continuous integration testing.

## Creating a Pull Request

If you are creating a PR for another repo, be mindful of their contributing guidelines and strive for clarity in your commits. Many projects have strict requirements for PRs and may provide a [detailed checklist](https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md#pull-request-checklist) that puts the onus on contributors to ensure that new code is necessary, compliant, and able to pass testing.

It's also a good idea to review or search the list of existing open PRs to see if someone else has already addressed the changes you're proposing. To open a new PR, navigate to the repo's Pull Requests page (`/pulls`) and click the green button. (Pushing your changes at the command line or via a desktop app will automatically initiate the comparison and present a green button on the repo's home page.)

**Compare:** Select your `fork:branch` to compare against `repo:main` (or default branch)

**Title:** Clearly worded statement or phrase (e.g., `Created a page that explains how to open PRs` or `New dependency graphs`)

**Body:**

* Describe the overall goal/effort that your PR addresses (e.g., `Fully reworks feature X to accommodate Windows users`)
* Describe what your commit(s) contain (e.g., `Added new page Y and updated README`)
* Mention/link to any related issues or PRs (e.g., `Closes issue #456` or `Fixes bug #789`)
* Mention (`@`) a specific person (e.g., a reviewer) if applicable
