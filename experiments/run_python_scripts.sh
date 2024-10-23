# This is a bash file for running 
# diffusion model hyperparameter 
# test, each time modifying a single
# parameter (either high or low

# We are testing feature channels (widths), 
# block depth, downsample blocks, and sinusoidal
# embeddings, and then a combination of both


# execute these jupyter notebook and save it in place

echo "Starting new script... (1)"
jupyter nbconvert --execute --inplace exp4_ddim_300.ipynb

echo "Starting new script... (2)"
jupyter nbconvert --execute --inplace exp5_ddim_300.ipynb

echo "Bash file finished executing!"
