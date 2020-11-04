# Data

http://bits.csb.pitt.edu/files/cnn-ani/

# Training example

`difftraining.py --activation_function=elu --clip=1 --filter_factor=2 --hidden_size=1024 --lr=0.1 --maxepoch=20 --module_connect=residual --module_depth=6 --module_filters=64 --module_kernel_size=3 --num_modules=5 --pickle=fulltraintest.pickle --resolution=0.5 --solver=sgd --stop=100000`


# Evaluating a trained model on a molecule

`./evalcnn.py colorful.pt linmodel.pickle molecules/dihedral/sucrose/xyz/sucrose_0.xyz`
