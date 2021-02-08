# Test conversion script

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 0 --ploidy 1

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 1 --ploidy 1

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 0 --ploidy 1 --lengths 245

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 1 --ploidy 1 --lengths 245

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 0 --ploidy 1 --bed_file invalid.bed

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 1 --ploidy 1 --bed_file invalid.bed

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 0 --ploidy 1 --lengths 245 --bed_file invalid.bed

python pastis_to_pdb.py --input_file test_script/haploid.coords --output_file test1.pdb --interpolate 1 --ploidy 1 --lengths 245 --bed_file invalid.bed

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 0 --ploidy 2

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --lengths 245

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --lengths 245 250

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --lengths 60 60

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --bed_file test_script/counts.bed

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --bed_file test_script/invalid.bed

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --lengths 245 --bed_file test_script/counts.bed

python pastis_to_pdb.py --input_file test_script/diploid.coords --output_file test1.pdb --interpolate 1 --ploidy 2 --lengths 60 60 --bed_file test_script/counts.bed

rm test1.pdb 
