import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch

def plot_cluster_example():
	# Here we predefine the parameters
	parameters_1a = [5, 7, 5, 7]
	parameters_1b = [5, 5, 6, 6]

	parameters_2a = [20, 22, 20, 22]
	parameters_2b = [20, 20, 21, 21]

	# Set up our plot
	fig, ax = plt.subplots(figsize=(10, 8))

	ax.plot(parameters_1a, parameters_1b, "o", color="green")
	ax.plot(parameters_2a, parameters_2b, "o", color="green")

	# Draw the 0th layer in hierarchy
	circle0 = patches.Ellipse((13.5, 13), 28, 6,
	                     angle=45, linewidth=2, fill=False, zorder=2)

	ax.add_patch(circle0)

	# Draw the 1st layer in hierarchy
	circle1 = plt.Circle((6, 5.5), 2, color='b', fill=False)
	circle2 = plt.Circle((21, 20.5), 2, color='b', fill=False)

	ax.add_artist(circle1)
	ax.add_artist(circle2)

	# Draw the 2nd layer in hierarchy
	circle3 = plt.Circle((5, 5.5), 1, color='r', fill=False)
	circle4 = plt.Circle((7, 5.5), 1, color='r', fill=False)
	circle5 = plt.Circle((20, 20.5), 1, color='r', fill=False)
	circle6 = plt.Circle((22, 20.5), 1, color='r', fill=False)

	ax.add_artist(circle3)
	ax.add_artist(circle4)
	ax.add_artist(circle5)
	ax.add_artist(circle6)

	ax.set_xlim(0, 25)
	ax.set_ylim(0, 25)
	ax.set_xlabel("$a$")
	ax.set_ylabel("$b$")

	plt.show()



class data_generator():
    def __init__(self, num_tasks, std_dev, cluster_centers, K=5):
        
        # Define relevant parameters
        self.num_tasks = num_tasks
        self.std_dev = std_dev
        self.cluster_centers = cluster_centers
        self.K = K
        
        # Range of x values used in y = ax + b to produce data
        self.range = (-20, 20)
        
        # Initialise lists to store the generated parameters
        self.parameter_a = []
        self.parameter_b = []
        
        # Initialise lists to store the cluster label for each initialised parameter
        self.labels = []
        
        # Parameter that will store the last randomly generates index from get_linear_data
        self.last_rnd_idx = None
        
        # Run generate parameters function on initialisation of object
        self.generate_parameters()
        
        
    def generate_parameters(self):
        for n, cluster in enumerate(self.cluster_centers):
            # Generate distributions for parameter a and b based on the predefined cluster centers
            dist_a = np.random.normal(cluster[0], self.std_dev, self.num_tasks)
            dist_b = np.random.normal(cluster[1], self.std_dev, self.num_tasks)
            
            # Append these parameters to the class level lists
            self.parameter_a.append(dist_a)
            self.parameter_b.append(dist_b)
            
            # Append the appropriate cluster labels for each parameter pair to the list of labels
            self.labels.append(n * np.ones(self.num_tasks))
    
    
    def plot_parameter_dist(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        for cluster_a, cluster_b in zip(self.parameter_a, self.parameter_b):
            ax.plot(cluster_a, cluster_b, ".")

        # Draw the 0th layer in hierarchy
        circle0 = patches.Ellipse((13.5, 13), 28, 6,
                         angle=45, linewidth=2, fill=False, zorder=2)

        ax.add_patch(circle0)

        # Draw the 1st layer in hierarchy
        circle1 = plt.Circle((6, 5.5), 2, color='b', fill=False)
        circle2 = plt.Circle((21, 20.5), 2, color='b', fill=False)

        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # Draw the 2nd layer in hierarchy
        circle3 = plt.Circle((5, 5.5), 1, color='r', fill=False)
        circle4 = plt.Circle((7, 5.5), 1, color='r', fill=False)
        circle5 = plt.Circle((20, 20.5), 1, color='r', fill=False)
        circle6 = plt.Circle((22, 20.5), 1, color='r', fill=False)

        ax.add_artist(circle3)
        ax.add_artist(circle4)
        ax.add_artist(circle5)
        ax.add_artist(circle6)

        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        ax.set_xlabel("$a$")
        ax.set_ylabel("$b$")

        plt.show()
        
        
    def get_linear_data(self, cluster_n, test=False):
    	# Create K random x points 
        x = np.random.uniform(self.range[0], self.range[1], self.K)
        
        # Pick a random idex to randomly pick some parameters from our parameter lists
        rnd_idx = np.random.randint(0, self.num_tasks)
        # Assign this random variable to a class variable so that we can access the last task
        if test == False:	
        	self.last_rnd_idx = rnd_idx
        
        # If test=True then generate data from the previous task
        if test:
            a, b = self.parameter_a[cluster_n][self.last_rnd_idx], self.parameter_b[cluster_n][self.last_rnd_idx]
            y = a * x + b
            
        # else generate data for the new task
        else:
            a, b = self.parameter_a[cluster_n][rnd_idx], self.parameter_b[cluster_n][rnd_idx]
            y = a * x + b
#             print(rnd_idx)
#             print(a,b)
        
        # Recast the data into tensors
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        return x, y
    
    
    def plot_example(self):
    	# Iterates through predefined cluster centers and plots some example data
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(len(self.cluster_centers)):
            x, y = self.get_linear_data(i)
            ax.plot(x, y, 'o-')
        plt.show()