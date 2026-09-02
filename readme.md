General Instructions for Git Commits

1. First reach the directory that you want to make the master

2. git init: makes the current directory master, which means that it will be used to add the files to the GitHub Repo.

3. git add {instructions}: this instruction makes the selected files ready to be pushed. Only the named files in the instructions slot are selected.

{instructions}> individual file names, * which means "all"
in case you want to deselect some particular file : create a file called .gitignore and add the name of the files in it that you do not want to add to the GitHub repo.

git commit -m "comment": this command commits, that is actually adds the selected files to the github repos

git add remote origin {webaddress of the repo}: links the repo with the local current master directory

* (if not done previously only then) git branch -M main: makes the master local directory linked with the repo under branch name "main"

git push -u origin main: Completes the push of the files from the local machine to the github repo.