# Test conversion script

# Haploid, no interpolate
python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test_script/test1.pdb --interpolate 0 --ploidy 1

python test_script.py --pastis_coords_file test_script/haploid.coords --pdb_file test_script/test1.pdb --interpolate 0 --ploidy 1

# Haploid, interpolate
python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test_script/test1.pdb --interpolate 1 --ploidy 1

python test_script.py --pastis_coords_file test_script/haploid.coords --pdb_file test_script/test1.pdb --interpolate 1 --ploidy 1

# Diploid, no interpolate
python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test_script/test1.pdb --interpolate 0 --ploidy 2

python test_script.py --pastis_coords_file test_script/diploid.coords --pdb_file test_script/test1.pdb --interpolate 0 --ploidy 2

# Diploid, interpolate
python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test_script/test1.pdb --interpolate 1 --ploidy 2 --bed_file test_script/counts.bed

python test_script.py --pastis_coords_file test_script/diploid.coords --pdb_file test_script/test1.pdb --interpolate 1 --ploidy 2

rm test_script/test1.pdb
