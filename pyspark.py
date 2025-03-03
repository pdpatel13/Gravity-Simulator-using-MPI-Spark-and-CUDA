from pyspark.sql import SparkSession
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time
import datetime
import os


@dataclass
class Particle:
   position: np.ndarray
   velocity: np.ndarray
   mass: float
  
   def to_dict(self):
       return {
           'position': self.position.tolist(),
           'velocity': self.velocity.tolist(),
           'mass': float(self.mass)
       }
  
   @staticmethod
   def from_dict(d):
       return Particle(
           position=np.array(d['position']),
           velocity=np.array(d['velocity']),
           mass=d['mass']
       )


def calculate_force_between(p1_data, p2_data, G):
   p1_pos = np.array(p1_data['position'])
   p2_pos = np.array(p2_data['position'])
   r = p2_pos - p1_pos
   distance = np.linalg.norm(r)
  
   if distance < 1e-10:
       return np.zeros(3).tolist()
      
   force_magnitude = (G * p1_data['mass'] * p2_data['mass']) / (distance ** 2)
   return (force_magnitude * r / distance).tolist()


class SparkGravitySimulator:
   G = 6.67430e-11
  
   def __init__(self, particles: List[Particle], dt: float = 0.01, cores: int = 1, memory: str = "2g"):
       self.spark = SparkSession.builder.master(f"local[{cores}]") \
           .appName("GravitySimulation") \
           .config("spark.executor.memory", memory) \
           .config("spark.executor.cores", "1") \
           .getOrCreate()
      
       self.particles_data = [p.to_dict() for p in particles]
       self.dt = dt
       self.num_particles = len(particles)
  
   def calculate_forces(self):
       particle_pairs = []
       for i in range(self.num_particles):
           for j in range(i + 1, self.num_particles):
               particle_pairs.append((i, j))
              
       sc = self.spark.sparkContext
       particles_broadcast = sc.broadcast(self.particles_data)
       G_broadcast = sc.broadcast(self.G)
      
       pairs_rdd = sc.parallelize(particle_pairs)
      
       def calculate_pair_force(pair):
           i, j = pair
           particles = particles_broadcast.value
           G = G_broadcast.value
           force = calculate_force_between(particles[i], particles[j], G)
           return (i, j, force)
      
       forces = pairs_rdd.map(calculate_pair_force).collect()
      
       total_forces = [np.zeros(3) for _ in range(self.num_particles)]
       for i, j, force in forces:
           force_array = np.array(force)
           total_forces[i] += force_array
           total_forces[j] -= force_array
          
       return total_forces
  
   def update(self):
       forces = self.calculate_forces()
      
       for i, particle_data in enumerate(self.particles_data):
           force = forces[i]
           mass = particle_data['mass']
           velocity = np.array(particle_data['velocity'])
           position = np.array(particle_data['position'])
          
           acceleration = force / mass
           velocity += acceleration * self.dt
           position += velocity * self.dt
          
           particle_data['velocity'] = velocity.tolist()
           particle_data['position'] = position.tolist()
  
   def run_simulation(self, steps: int) -> List[List[Tuple[float, float, float]]]:
       trajectories = [[] for _ in range(self.num_particles)]
      
       start_time = time.time()
       for step in range(steps):
           if step % 100 == 0:
               print(f"Step {step}/{steps}")
          
           self.update()
          
           for i, particle_data in enumerate(self.particles_data):
               trajectories[i].append(tuple(particle_data['position']))
      
       end_time = time.time()
       print(f"Simulation took {end_time - start_time:.2f} seconds")
      
       self.spark.stop()
       return trajectories


def create_solar_system() -> List[Particle]:
   return [
       Particle(
           position=np.array([0.0, 0.0, 0.0]),
           velocity=np.array([0.0, 0.0, 0.0]),
           mass=1.989e30
       ),
       Particle(
           position=np.array([1.496e11, 0.0, 0.0]),
           velocity=np.array([0.0, 29.78e3, 0.0]),
           mass=5.972e24
       ),
       Particle(
           position=np.array([2.279e11, 0.0, 0.0]),
           velocity=np.array([0.0, 24.077e3, 0.0]),
           mass=6.39e23
       )
   ]


def generate_random_particles(num_particles: int) -> List[Particle]:
   return [Particle(
           position=np.random.uniform(-3e11, 3e11, 3),
           velocity=np.random.uniform(-30e3, 30e3, 3),
           mass=np.random.uniform(1e23, 1e25)
       ) for _ in range(num_particles)]


if __name__ == "__main__":
   if not os.path.exists('gravity_logs_spark'):
       os.makedirs('gravity_logs_spark')


   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   log_file = f"gravity_logs_spark/simulation_log_{timestamp}.txt"
  
   def log_print(message):
       print(message)
       with open(log_file, 'a') as f:
           f.write(message + '\n')


   base_particles = create_solar_system()
  
   configurations = [
       {'cores': 2, 'particles': 10},
       {'cores': 2, 'particles': 100},
       {'cores': 2, 'particles': 500},
       {'cores': 2, 'particles': 1000}
   ]

   for config in configurations:
       cores = config['cores']
       num_particles = config['particles']

       particles = base_particles + generate_random_particles(num_particles - len(base_particles))

       log_print(f"\nStarting gravity simulation with {cores} cores and {num_particles} particles")
       log_print("Configuration:")
       log_print(f"- Number of steps: 500")
       log_print(f"- Time step: 3600 seconds (1 hour)")

       simulator = SparkGravitySimulator(particles, dt=3600, cores=cores, memory="4g")
       time_start = time.time()
       trajectories = simulator.run_simulation(steps=500)

       total_time = time.time() - time_start
       log_print("\nPerformance Statistics:")
       log_print(f"Total execution time: {total_time:.2f} seconds")
       log_print(f"Average time per step: {total_time/500:.4f} seconds")
  
       log_print("\nFinal positions:")
       for i, trajectory in enumerate(trajectories):
           final_pos = trajectory[-1]
           log_print(f"Particle {i}: {final_pos}")

   log_print("\nSimulation completed successfully")
