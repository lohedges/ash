#!/bin/bash

#other options:
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself/246128#246128
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "DIR is $DIR"

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
echo "DIT : $DIT"

#echo $(dirname "${BASH_SOURCE{0}}")
#echo "bash source : $(dirname "${BASH_SOURCE[0]}")"

#Getting path of this shell-script (resids in Yggdrasill main dir)
SCRIPT=$(readlink -f "$0")
echo "SCRIPT : $SCRIPT"


yggdrasill_path=$(dirname "$SCRIPT")

echo "Setting PYTHONPATH and LD_LIBRARY_PATH for Yggdrasill"
#PYTHONPATH
export PYTHONPATH=$yggdrasill_path:$PYTHONPATH
#Setting LD_LIBRARY_PATH for lib dir
export LD_LIBRARY_PATH=$yggdrasill_path/lib:$LD_LIBRARY_PATH