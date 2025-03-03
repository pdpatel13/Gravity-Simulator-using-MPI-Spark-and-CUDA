#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <random>
#include <cuda_runtime.h>
#include <iomanip>

// Constants
const float G = 6.67430e-11f; // Gravitational constant

// Particle Class using raw arrays
struct Particle {
    float position[3]; // Position in 3D space
    float velocity[3]; // Velocity in 3D space
    float mass;        // Mass of the particle

    // Constructor for Particle
    Particle(float pos[3], float vel[3], float m) : mass(m) {
        std::copy(pos, pos + 3, position);
        std::copy(vel, vel + 3, velocity);
    }

    Particle() : mass(0.0f) {
        std::fill(position, position + 3, 0.0f);
        std::fill(velocity, velocity + 3, 0.0f);
    }
};

// CUDA kernel to calculate gravitational forces
__device__ void calculate_force_between(const Particle* particles, int i, int j, float* forces) {
    float dx = particles[j].position[0] - particles[i].position[0];
    float dy = particles[j].position[1] - particles[i].position[1];
    float dz = particles[j].position[2] - particles[i].position[2];

    float r = sqrt(dx * dx + dy * dy + dz * dz);

    if (r < 1e-10f) return;

    float force_magnitude = (G * particles[i].mass * particles[j].mass) / (r * r);

    forces[3 * i] += force_magnitude * dx / r;
    forces[3 * i + 1] += force_magnitude * dy / r;
    forces[3 * i + 2] += force_magnitude * dz / r;

    forces[3 * j] -= force_magnitude * dx / r;
    forces[3 * j + 1] -= force_magnitude * dy / r;
    forces[3 * j + 2] -= force_magnitude * dz / r;
}

// CUDA Kernel to calculate forces
__global__ void calculate_forces_kernel(const Particle* particles, int num_particles, float* forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    for (int j = i + 1; j < num_particles; ++j) {
        calculate_force_between(particles, i, j, forces);
    }
}

// Function to update positions and velocities based on forces
void update(std::vector<Particle>& particles, std::vector<float>& forces, float dt) {
    for (int i = 0; i < particles.size(); ++i) {
        float* force = &forces[3 * i];
        Particle& p = particles[i];

        float acceleration[3];
        for (int j = 0; j < 3; ++j) {
            acceleration[j] = force[j] / p.mass;
        }

        for (int j = 0; j < 3; ++j) {
            p.velocity[j] += acceleration[j] * dt;
            p.position[j] += p.velocity[j] * dt;
        }
    }
}

// Function to create a solar system with particles
std::vector<Particle> create_solar_system() {
    float sun_pos[3] = {0.0f, 0.0f, 0.0f};
    float sun_vel[3] = {0.0f, 0.0f, 0.0f};
    std::vector<Particle> particles;
    particles.push_back(Particle(sun_pos, sun_vel, 1.989e30f)); // Sun

    float earth_pos[3] = {1.496e11f, 0.0f, 0.0f};
    float earth_vel[3] = {0.0f, 29.78e3f, 0.0f};
    particles.push_back(Particle(earth_pos, earth_vel, 5.972e24f)); // Earth

    float mars_pos[3] = {2.279e11f, 0.0f, 0.0f};
    float mars_vel[3] = {0.0f, 24.077e3f, 0.0f};
    particles.push_back(Particle(mars_pos, mars_vel, 6.39e23f)); // Mars

    return particles;
}

// Function to print the final positions of the particles
void print_final_positions(const std::vector<Particle>& particles, int num_particles, int print_particles) {
    std::cout << "\nFinal positions of first " << print_particles << " out of " << num_particles << " particles:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        const Particle& p = particles[i];
        std::cout << "Particle " << i << ": ("
                  << std::fixed << std::setprecision(14)
                  << p.position[0] << ", "
                  << p.position[1] << ", "
                  << p.position[2] << ")"
                  << std::endl;
    }
}

// Logging function
void log_print(const std::string& message, const std::string& log_file) {
    std::cout << message << std::endl;
    std::ofstream log(log_file, std::ios::app);
    log << message << std::endl;
}

// Main simulation function
int main() {
    const int num_particles = 50000; // Example number of particles
    const int print_particles = 10; // Number of particles to print
    const float dt = 3600.0f; // Time step in seconds

    std::vector<Particle> particles = create_solar_system();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-3e11f, 3e11f);
    std::uniform_real_distribution<float> vel_dist(-30e3f, 30e3f);
    std::uniform_real_distribution<float> mass_dist(1e23f, 1e25f);

    for (int i = 0; i < num_particles - 3; ++i) {
        float pos[3] = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
        float vel[3] = {vel_dist(gen), vel_dist(gen), vel_dist(gen)};
        float mass = mass_dist(gen);
        particles.push_back(Particle(pos, vel, mass));
    }

    auto timestamp = std::chrono::system_clock::now();
    std::time_t timestamp_time = std::chrono::system_clock::to_time_t(timestamp);
    std::string log_file = "gravity_logs_spark/simulation_log_" + std::to_string(timestamp_time) + ".txt";
    log_print("Starting gravity simulation", log_file);

    Particle* d_particles;
    float* d_forces;

    cudaMalloc(&d_particles, particles.size() * sizeof(Particle));
    cudaMalloc(&d_forces, 3 * particles.size() * sizeof(float));

    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemset(d_forces, 0, 3 * particles.size() * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 500; ++step) {
        calculate_forces_kernel<<<(particles.size() + 255) / 256, 256>>>(d_particles, particles.size(), d_forces);
        cudaDeviceSynchronize();

        std::vector<float> forces(3 * particles.size());
        cudaMemcpy(forces.data(), d_forces, 3 * particles.size() * sizeof(float), cudaMemcpyDeviceToHost);

        update(particles, forces, dt);

        if (step % 100 == 0) {
            log_print("Step " + std::to_string(step), log_file);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    log_print("Simulation took " + std::to_string(duration.count()) + " seconds", log_file);

    print_final_positions(particles, num_particles, print_particles);

    log_print("Simulation completed successfully", log_file);

    return 0;
}
