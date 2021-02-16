cd $1
for f in *; do
	cd $f
	printf "$(pwd)"
	cd ..
	printf "\n"
done
cd /home/ruffian/Documents/Research/Rey/Penning/ion-trapping-notes/
