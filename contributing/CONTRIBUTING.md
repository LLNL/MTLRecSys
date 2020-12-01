---
title: "Contributing Guidelines"
---

*Contributing guidelines* communicate how people should contribute to your repo. The file is typically named in all-caps: `CONTRIBUTING.md`.

There's no single format for this type of file. Some examples of the range of guidelines in LLNL repos include [Umpire](https://github.com/LLNL/Umpire/blob/develop/CONTRIBUTING.md), [SUNDIALS](https://github.com/LLNL/sundials/blob/master/CONTRIBUTING.md), and [MFEM](https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md). An example of a variation across repos is the name of the branch contributors should create their own branches from (e.g., `main`, `master`, `develop`). Some projects include ancillary documentation alongside or within the `CONTRIBUTING.md` file such as a [detailed checklist](https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md#pull-request-checklist) for PRs, an [explanation](https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md#code-overview) of source code directory structure, or a [coding style guide](https://flux-framework.readthedocs.io/projects/flux-rfc/en/latest/spec_7.html) to encourage uniformity.

The outline below contains generic sample language as well as a few blanks `_____` where you fill in your project/repo/team's name and/or email address. For more information, see GitHub's tips for [setting guidelines for repository contributors](https://docs.github.com/en/free-pro-team@latest/github/building-a-strong-community/setting-guidelines-for-repository-contributors).

As you create your contributor guidelines, consider the associated logistics, file organization, naming conventions, and other similar details. For example, what information should each pull request or issue include? How do you want bugs reported? What about testing? Just as an email with a vague subject line is frustrating and unhelpful, so too is a PR with poorly named commits. Another analogy: Just as you don't know what to expect in a meeting invitation without an agenda, the same goes for a bug that's reported without enough information to recreate the problem.

## Example Contributing Guidelines

`_____` is an open source project. We welcome contributions via pull requests as well as questions, feature requests, or bug reports via issues. Contact our team at `_____` with any questions. Please also refer to our [code of conduct](https://github.com/LLNL/.github/tree/master/community-health/CODE_OF_CONDUCT.md).

If you aren't a `_____` developer at LLNL, you won't have permission to push new branches to the repository. First, you should create a fork. This will create your copy of the `_____` repository and ensure you can push your changes up to GitHub and create PRs. `_____` uses Travis for continuous integration tests. Our tests are automatically run against every new PR, and passing all tests is a requirement for merging your PR.

* Create your branches off the `repo:main` branch.
* Clearly name your branches, commits, and PRs as this will help us manage queued work in a timely manner.
* Articulate your commit messages in the imperative (e.g., `Adds new privacy policy link to README`).
* Commit your work in logically organized commits, and group commits together logically in a PR.
* Title each PR clearly and give it an unambiguous description.
* Review existing issues before opening a new one. Your issue might already be under development or discussed by others. Feel free to add to any outstanding issue/bug.
* Be explicit when opening issues and reporting bugs. What behavior are you expecting? What is your justification or use case for the new feature/enhancement? How can the bug be recreated? What are any environment variables to consider?
