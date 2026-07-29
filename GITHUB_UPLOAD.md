# Uploading this folder to GitHub

The folder is already structured as a repository. Using GitHub Desktop or Git from a terminal is recommended because the repository contains thousands of small label files and four binary checkpoints.

## Command-line upload

Create an empty repository on GitHub without automatically adding a README, then run from this folder:

```powershell
git init
git add .
git commit -m "Add wildfire smoke location benchmark"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL.

## Pre-upload checklist

- Replace project-owner placeholders if you add a code license or repository citation.
- Confirm that the PDF and sample figures display correctly.
- Do not add the excluded raw-image or resized-image directories.
- Keep the dataset attribution and CC BY 4.0 notice.
- Review checkpoint distribution against the pretrained-model license that applies at upload time.

Every included file is below GitHub's 25 MiB browser-upload limit and 100 MiB regular-Git hard limit, but Git or GitHub Desktop will handle the many annotation files more reliably than the browser uploader.

