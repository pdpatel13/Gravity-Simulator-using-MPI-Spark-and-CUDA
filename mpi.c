#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#define G 6.67430e-11

typedef struct {
    double x;
    double y;
    double z;
} Vector3;

typedef struct {
    Vector3 position;
    Vector3 velocity;
    double mass;
} Particle;

Vector3 vector3_zero() {
    Vector3 v = {0.0, 0.0, 0.0};
    return v;
}

Vector3 vector3_subtract(Vector3 a, Vector3 b) {
    Vector3 result = {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
    return result;
}

Vector3 vector3_multiply(Vector3 v, double scalar) {
    Vector3 result = {
        v.x * scalar,
        v.y * scalar,
        v.z * scalar
    };
    return result;
}

Vector3 vector3_add(Vector3 a, Vector3 b) {
    Vector3 result = {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
    return result;
}

double vector3_magnitude(Vector3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3 calculate_force(Particle* p1, Particle* p2) {
    Vector3 force = vector3_zero();
    Vector3 r = vector3_subtract(p2->position, p1->position);
    double distance = vector3_magnitude(r);
    
    if (distance < 1e-10) {
        return force;
    }
    
    double force_magnitude = (G * p1->mass * p2->mass) / (distance * distance);
    double scale = force_magnitude / distance;
    force = vector3_multiply(r, scale);
    
    return force;
}

void create_solar_system(Particle* particles, int* num_particles) {
    particles[0].position = vector3_zero();
    particles[0].velocity = vector3_zero();
    particles[0].mass = 1.989e30;
    
    particles[1].position.x = 1.496e11;
    particles[1].position.y = 0.0;
    particles[1].position.z = 0.0;
    particles[1].velocity.x = 0.0;
    particles[1].velocity.y = 29.78e3;
    particles[1].velocity.z = 0.0;
    particles[1].mass = 5.972e24;
    
    particles[2].position.x = 2.279e11;
    particles[2].position.y = 0.0;
    particles[2].position.z = 0.0;
    particles[2].velocity.x = 0.0;
    particles[2].velocity.y = 24.077e3;
    particles[2].velocity.z = 0.0;
    particles[2].mass = 6.39e23;
    
    srand(time(NULL));
    for (int i = 3; i < 8; i++) {
        particles[i].position.x = ((double)rand() / RAND_MAX) * 6e11 - 3e11;
        particles[i].position.y = ((double)rand() / RAND_MAX) * 6e11 - 3e11;
        particles[i].position.z = ((double)rand() / RAND_MAX) * 6e11 - 3e11;
        particles[i].velocity.x = ((double)rand() / RAND_MAX) * 6e4 - 3e4;
        particles[i].velocity.y = ((double)rand() / RAND_MAX) * 6e4 - 3e4;
        particles[i].velocity.z = ((double)rand() / RAND_MAX) * 6e4 - 3e4;
        particles[i].mass = ((double)rand() / RAND_MAX) * 9.9e24 + 1e23;
    }
    
    *num_particles = 8;
}

void create_log_file(char* filename, int num_processes, int num_particles, int steps, double dt) {
    struct stat st = {0};
    if (stat("gravity_logs_mpi", &st) == -1) {
        #ifdef _WIN32
        mkdir("gravity_logs_mpi");
        #else
        mkdir("gravity_logs_mpi", 0700);
        #endif
    }

    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm);
    
    sprintf(filename, "gravity_logs_mpi/mpi_c_simulation_%s.txt", timestamp);
    
    FILE* file = fopen(filename, "w");
    if (file != NULL) {
        fprintf(file, "Starting MPI C gravity simulation at %s\n", timestamp);
        fprintf(file, "Number of processes: %d\n", num_processes);
        fprintf(file, "Number of particles: %d\n", num_particles);
        fprintf(file, "Steps: %d\n", steps);
        fprintf(file, "Timestep: %f seconds\n\n", dt);
        fclose(file);
    } else {
        printf("Error: Could not create log file\n");
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int MAX_PARTICLES = 100;
    const int STEPS = 500;
    const double DT = 3600.0;
    
    Particle* all_particles = NULL;
    int num_particles = 0;
    char log_filename[256];
    
    if (rank == 0) {
        all_particles = (Particle*)malloc(MAX_PARTICLES * sizeof(Particle));
        create_solar_system(all_particles, &num_particles);
        create_log_file(log_filename, size, num_particles, STEPS, DT);
    }
    
    MPI_Bcast(&num_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        all_particles = (Particle*)malloc(num_particles * sizeof(Particle));
    }
    
    MPI_Datatype particle_type;
    int blocklengths[] = {3, 3, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    MPI_Get_address(&all_particles[0].position, &displacements[0]);
    MPI_Get_address(&all_particles[0].velocity, &displacements[1]);
    MPI_Get_address(&all_particles[0].mass, &displacements[2]);
    
    for (int i = 2; i >= 0; i--) {
        displacements[i] = MPI_Aint_diff(displacements[i], displacements[0]);
    }
    
    MPI_Type_create_struct(3, blocklengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);
    
    MPI_Bcast(all_particles, num_particles, particle_type, 0, MPI_COMM_WORLD);
    
    int particles_per_proc = num_particles / size;
    int remainder = num_particles % size;
    int start_idx = rank * particles_per_proc + (rank < remainder ? rank : remainder);
    int local_num_particles = particles_per_proc + (rank < remainder ? 1 : 0);
    
    double start_time = MPI_Wtime();
    
    for (int step = 0; step < STEPS; step++) {
        if (rank == 0 && step % 100 == 0) {
            printf("Step %d/%d\n", step, STEPS);
        }
        
        for (int i = start_idx; i < start_idx + local_num_particles; i++) {
            Vector3 total_force = vector3_zero();
            
            for (int j = 0; j < num_particles; j++) {
                if (i != j) {
                    Vector3 force = calculate_force(&all_particles[i], &all_particles[j]);
                    total_force = vector3_add(total_force, force);
                }
            }
            
            Vector3 acceleration = vector3_multiply(total_force, 1.0 / all_particles[i].mass);
            all_particles[i].velocity = vector3_add(
                all_particles[i].velocity,
                vector3_multiply(acceleration, DT)
            );
            
            all_particles[i].position = vector3_add(
                all_particles[i].position,
                vector3_multiply(all_particles[i].velocity, DT)
            );
        }
        
        int *recvcounts = (int*)malloc(size * sizeof(int));
        int *displs = (int*)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            int particles_for_proc = particles_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = particles_for_proc;
            displs[i] = (i > 0) ? displs[i-1] + recvcounts[i-1] : 0;
        }

        MPI_Allgatherv(
            &all_particles[start_idx], local_num_particles, particle_type,
            all_particles, recvcounts, displs, particle_type,
            MPI_COMM_WORLD
        );

        free(recvcounts);
        free(displs);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    
    if (rank == 0) {
        FILE* file = fopen(log_filename, "a");
        if (file != NULL) {
            fprintf(file, "\nPerformance Statistics:\n");
            fprintf(file, "Total execution time: %.2f seconds\n", total_time);
            fprintf(file, "Average time per step: %.4f seconds\n", total_time/STEPS);
            
            fprintf(file, "\nFinal positions:\n");
            for (int i = 0; i < num_particles; i++) {
                fprintf(file, "Particle %d: (%e, %e, %e)\n",
                    i,
                    all_particles[i].position.x,
                    all_particles[i].position.y,
                    all_particles[i].position.z
                );
            }
            
            fprintf(file, "\nSimulation completed successfully\n");
            fclose(file);
        }
    }
    
    MPI_Type_free(&particle_type);
    free(all_particles);
    MPI_Finalize();
    
    return 0;
}