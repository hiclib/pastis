# Test conversion script

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test_script/test1.pdb --interpolate 0 --ploidy 1

python test_script.py --pastis_coords_file test_script/haploid.coords --pdb_file test_script/test1.pdb

rm test_script/test1.pdb 
