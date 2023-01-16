import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Create a neural network model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Create dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Swarm Intelligence

# Define Agent class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

class Agent(creator.Particle):
    def __init__(self, decision_making_ability, model, data, agents):
        super(Agent, self).__init__()
        self.decision_making_ability = decision_making_ability
        self.model = model
        self.data = data
        self.loss = 0
        self.agents = agents

    def communicate(self):
        weights = self.get_weights()
        for other in self.agents:
            if other != self:
                other.set_weights(weights)

    def update_model(self):
        self.model.fit(self.data[0], self.data[1], epochs=1, batch_size=32)
        self.loss = self.model.evaluate(self.data[2], self.data[3], verbose=0)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

# Create agents
agents = []
agents = [Agent(decision_making_ability, model, (X_train, to_categorical(y_train), X_test, to_categorical(y_test)), agents) for decision_making_ability in range(10)]

# Train the model
for i in range(10):
    for agent in agents:
        agent.update_model()
        agent.communicate()


# Optimization Techniques
# Define optimization function


inertia = 0.5 # starting value for inertia
inertia_decay = 0.9 # decay rate for inertia
c1 = 1 # starting value for c1
c1_decay = 0.99 # decay rate for c1
c2 = 2 # starting value for c2
c2_decay = 0.99 # decay rate for c2
n_iterations_without_improvement = 10 # number of iterations without improvement before decay
particles = agents
def PSO(particles, n_iterations):
    # Initialize variables
    n_particles = len(particles)
    dimensions = len(particles[0].get_weights())

    position = [list(p) for p in particles]
    velocity = [[0 for _ in range(dimensions)] for _ in range(n_particles)]

    global_best_agent = max(particles, key=lambda x: x.loss)
    
    for i in range(n_iterations):
        for j, particle in enumerate(particles):
            # Update velocity
            for k in range(dimensions):
                velocity[j][k] = inertia * velocity[j][k] + c1 * random.random() * (global_best_agent.get_weights()[k] - position[j][k]) + c2 * random.random() * (particles[j][k] - position[j][k])

            # Update position
            for k in range(dimensions):
                position[j][k] += velocity[j][k]

            # Update personal best
            if particle.loss < particle.best:
                particle.best = particle.loss
                particle.best_position = list(particle)

            # Update global best
            if particle.loss < global_best_agent.loss:
                global_best_agent = particle

        # Update inertia, c1, and c2
        if i % n_iterations_without_improvement == 0:
            inertia *= inertia_decay
            c1 *= c1_decay
            c2 *= c2_decay



# Run swarm optimization
global_best = tools.selBest(particles, k=1, fit_attr='fitness')[0]
PSO(global_best, agents)
best_agent = tools.selBest(agents, k=1)[0]
print("Best agent's accuracy: ", best_agent.loss)


# Train the model
for i in range(10):
    for agent in agents:
        agent.update_model()
        agent.communicate()

# Evaluate the final model
_, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# Visualize the final model
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot()
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

plt.show()