echo "Searching for python3 in PATH"
path_to_python3_exe=$(which python3)

#Dirname only
path_to_python3_dir=${path_to_python3_exe%/python3}
echo ""
echo "Python3 path: $path_to_python3_dir"
echo ""

#The parent dir containing the ash dir
ash_parent_dir=$(dirname "$PWD")
ash_dir=$PWD


#Create set_environment_ash.sh file
echo "Creating set_environent_ash.sh script"
echo "#!/bin/bash" > set_environment_ash.sh
echo "ulimit -s unlimited" >> set_environment_ash.sh
echo "export ASHPATH=${ash_parent_dir}" >> set_environment_ash.sh
echo "export python3path=${path_to_python3_dir}" >> set_environment_ash.sh
echo "export PYTHONPATH=\$ASHPATH:\$ASHPATH/ash/lib:\$PYTHONPATH" >> set_environment_ash.sh
echo "export PATH=\$python3path:\$ASHPATH:\$PATH" >> set_environment_ash.sh
echo "export LD_LIBRARY_PATH=\$ASHPATH/ash/lib:\$LD_LIBRARY_PATH" >> set_environment_ash.sh


echo "Installation of ASH was successful!"
echo ""
echo "Remember:"
echo "     - Run: source ${thisdir}/set_environment_ash.sh to activate ASH!"
echo "     - Put source command in your .bash_profile/.zshrc/.bashrc and job-submission scripts"
