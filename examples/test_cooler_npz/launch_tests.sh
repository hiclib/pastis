# DIPLOID UNAMBIGUOUS
# Convert cooler file to npz
cooler-to-npz --lengths cooler_example.bed --output_file cooler_example.npz \
--maternal_maternal cooler_example.cool --paternal_paternal cooler_example.cool

# Load the npz file
python load_npz.py

# Remove the generated file
rm cooler_example.npz

# DIPLOID FULLY AMBIGUOUS
# Convert cooler file to npz
cooler-to-npz --lengths cooler_example.bed --output_file cooler_example.npz \
--unknown_unknown cooler_example.cool

# Load the npz file
python load_npz.py

# Remove the generated file
rm cooler_example.npz

# HAPLOID
cooler-to-npz --lengths cooler_example.bed --output_file cooler_example.npz \
--haploid cooler_example.cool

# Load the npz file
python load_npz.py

# Remove the generated file
rm cooler_example.npz
