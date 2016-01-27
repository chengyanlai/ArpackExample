#/bin/bash
echo "to " $1
if [[ $1 == kagome ]]; then
  rsync -avzhe ssh \
  --include-from 'sync-include.txt' \
  --exclude-from 'sync-exclude.txt' \
  /Volumes/Files/GitRepo/ArpackExample $1:/condensate1/GitRepo/
else
  rsync -avzhe ssh \
  --include-from 'sync-include.txt' \
  --exclude-from 'sync-exclude.txt' \
  /Volumes/Files/GitRepo/ArpackExample $1:~/GitRepo/
fi
