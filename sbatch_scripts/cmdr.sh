cd Playground
for i in {1..3}; do
for j in {1..3}; do
if [ $i -ne $j ]
then
cp tmax10e-3_dt1e-9_md47_p99.run tmax10e-3_dt1e-9_md${i}${j}_p99.run
fi
done
done
cd ..
