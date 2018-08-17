cd $1
for f in *; do
	cd $f
	cur_dir="$PWD/"
	python /home/ruffian/Documents/Research/Rey/Penning/ion-trapping-notes/plot_evolution.py $cur_dir
	#printf $cur_dir
	#printf "\n"
	cd ..
done
cd /home/ruffian/Documents/Research/Rey/Penning/ion-trapping-notes/
