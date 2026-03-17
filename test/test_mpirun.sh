mpirun \
    --allow-run-as-root \
    --hostfile test/hostfile.txt \
    --map-by slot \
    --mca enable eth0 \
    bash /nvme/wht/code/blitz-remake/test/test.sh
