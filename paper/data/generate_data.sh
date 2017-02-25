mkdir -p ./basement/grid/trial1 ./basement/grid/trial2 ./basement/grid/trial3 ./basement/grid/trial4 ./basement/grid/trial5
mkdir -p ./synthetic/grid/trial1 ./synthetic/grid/trial2 ./synthetic/grid/trial3 ./synthetic/grid/trial4 ./synthetic/grid/trial5
mkdir -p ./basement/random/trial1 ./basement/random/trial2 ./basement/random/trial3 ./basement/random/trial4 ./basement/random/trial5
mkdir -p ./synthetic/random/trial1 ./synthetic/random/trial2 ./synthetic/random/trial3 ./synthetic/random/trial4 ./synthetic/random/trial5
mkdir -p ./basement/particle/trial1 ./basement/particle/trial2 ./basement/particle/trial3 ./basement/particle/trial4 ./basement/particle/trial5
mkdir -p ./synthetic/particle/trial1 ./synthetic/particle/trial2 ./synthetic/particle/trial3 ./synthetic/particle/trial4 ./synthetic/particle/trial5

# # populate the serialized datastructures for visualization
../../build/bin/range_lib --method="cddt" --map_path=./basement_hallways_5cm.png --cddt_save_path=./basement/cddt.json
../../build/bin/range_lib --method="pcddt" --map_path=./basement_hallways_5cm.png --cddt_save_path=./basement/pcddt.json
../../build/bin/range_lib --method="cddt" --map_path=./synthetic.png --cddt_save_path=./synthetic/cddt.json
../../build/bin/range_lib --method="pcddt" --map_path=./synthetic.png --cddt_save_path=./synthetic/pcddt.json

# # grid benchmark for both maps/all methods
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/grid/trial1/ --which_benchmark=grid
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/grid/trial2/ --which_benchmark=grid
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/grid/trial3/ --which_benchmark=grid

../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/grid/trial1/ --which_benchmark=grid
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/grid/trial2/ --which_benchmark=grid
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/grid/trial3/ --which_benchmark=grid

# # random benchmark for both maps/all methods
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/random/trial1/ --which_benchmark=random
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/random/trial2/ --which_benchmark=random
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./basement_hallways_5cm.png --log_path=./basement/random/trial3/ --which_benchmark=random

../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/random/trial1/ --which_benchmark=random
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/random/trial2/ --which_benchmark=random
../../build/bin/range_lib --method="rm,cddt,pcddt,bl,glt" --map_path=./synthetic.png --log_path=./synthetic/random/trial3/ --which_benchmark=random

# particle filter benchmark
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial1/cddt.csv --method="cddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial1/pcddt.csv --method="pcddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial1/glt.csv --method="glt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial1/bl.csv --method="bl" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial1/rm.csv --method="rm" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000

../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial2/cddt.csv --method="cddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial2/pcddt.csv --method="pcddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial2/glt.csv --method="glt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial2/bl.csv --method="bl" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial2/rm.csv --method="rm" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000

../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial3/cddt.csv --method="cddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial3/pcddt.csv --method="pcddt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial3/glt.csv --method="glt" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial3/bl.csv --method="bl" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000
../../../particle_filter/build/bin/particle_filter --log_path=./basement/particle/trial3/rm.csv --method="rm" --sampling="standard" --map_path=./basement_hallways_5cm.png --max_particles=6000


# ../../../particle_filter/build/bin/particle_filter --viz --method="cddt" --sampling="kld" --map_path=./basement_hallways_5cm.png
