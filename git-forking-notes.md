# Some notes on working with forks of repositories

The most idiomatic work flow for collaborating on github is via
pull requests. The advantage of the pull request workflow is that
multiple developers can work on different features at their own
pace and they can manage their repositories without needing write
access to some central repository. They always commit and push to
their own repositories and when they would like to share their
work they do so via pull requests to another repository. These
notes give a quick overview of how to set things up and how to
get started.


## Forking a repository

The simplest way to fork a repository is via github's web
interface (we'll only talk about github hosted repositories
here). Simply click on the "fork" button on the website of the
repository you'd like to fork. This creates a copy of the
repository on the github web server. You can use this copy to
synchronize your work if you happen to work on multiple
computers.


## Cloning a git repository

Once you have a fork you can clone it to your local computer
(e.g. laptop) so you can make changes to the files, work with
programs, etc. For example, you can use the following commands to
clone your fork of the `ion-trapping-notes` repository:

```shell
git clone https://<github-username>@github.com/<github-username>/ion-trapping-notes
```

It is useful to add the `<github-username>@...` to the URL
because this will make sure that we automatically use the right
user account when we try to push to the remote repository.


## Pushing changes to the remote repository

Once you've made a few changes and committed them to your local
repository (the cloned repository) you can push them to the
remote so you or other folks can see the changes. To do that you
use the following command:

```shell
git push
```

Once you have the changes on github you can open a pull request
via the web interface.


## Synchronizing with an "upstream" repository

Typically you'll want to stay in sync with the "upstream"
repository, i.e. the original repository you forked. The easiest
way to do that is to add a second remote repository to your local
clone. You won't be able to push to that remote repository
because you typically won't have permissions to do so. But you
can pull, meaning that you incorporate the changes in the
"upstream" repository in your clone.

To add the upstream as an additional remote use the following
command:

```shell
git remote add upstream https://github.com/d-meiser/ion-trapping-notes 
```

To pull changes from the upstream repository you use the
following command:

```shell
git pull upstream
```

Typically you'll run this command whenever you want to get the
latest changes from upstream. There are some variations on this
command (such as first fetching and then merging) but in most
cases this is sufficient.

Once you've integrated the upstream changes you can push them to
your fork using `git push`.


## Resources

- Google is your friend!
- https://help.github.com for github specific topics. This page
  also has tons of useful recipes with ready to use code
  snippets.
- https://danielmiessler.com/study/git/ has a good git primer.
- There's a whole book on git available for free:
  https://git-scm.com/book/en/v2 This has tons of really good
  material if you'd like to understand a topic in more depth.
