/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::default_random_engine;
using std::normal_distribution;
using std::string;
using std::vector;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if (is_initialized) {
    return;
  }
  
  // Set the number of particles
  num_particles = 50;
  
  // Calculate normal distribution for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize every particle:
  // id = i
  // x, y, theta = (initial position + GPS uncertainty + random Gaussian noise)
  // weight = 1
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    
    particles.push_back(particle);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Go through each particle and add a measurement
  for (int i = 0; i < num_particles; ++i) {
    // Calculate new state based on derived theta
    // only if change in yaw_rate is significant
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) / yaw_rate;
      particles[i].y += velocity * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) / yaw_rate;
    }
    particles[i].theta += yaw_rate * delta_t;
    
    // Calculate normal distribution for x, y, theta
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    
    // Add random Gaussian noise to all predictions
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
   
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // Loop through each observation
  for (unsigned int i = 0; i < observations.size(); ++i) {
    
    // Initial distance is set to be the maximum of double
    double previous_dist = std::numeric_limits<double>::max();
    
    int closest_landmark_id;
    
     // Loop through each prediction
    for (unsigned int j = 0; j < predicted.size(); ++j) {
      
      // Calculate eucleadian distance between observation and prediction
      double euclidean_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      // If the distance is smaller than previous_dist,
      // set previous_dist as the new limit and pick current
      // predicted landmark as the closest to observation
      if (euclidean_dist < previous_dist) {
        previous_dist = euclidean_dist;
        closest_landmark_id = predicted[j].id;
      }
    }
    
    // Update observations id
    observations[i].id = closest_landmark_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
   // Go through each particle
   for (int i = 0; i < num_particles; ++i) {
   
     vector<LandmarkObs> landmarks_within_range;
     double sensor_range_area = sensor_range * sensor_range;
     
     // Go through each landmark
     for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
       
       // If the distance between particle and landmark is within sensor range area
       // preserve the landmark
       if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range_area) {
         landmarks_within_range.push_back(LandmarkObs{
           map_landmarks.landmark_list[j].id_i,
           map_landmarks.landmark_list[j].x_f,
           map_landmarks.landmark_list[j].y_f
         });
       }
     }
     
     // Apply 2D homogenous transformation from vehicle coordinates to map coordinates
     vector<LandmarkObs> transformed_observations;
     for (unsigned int j = 0; j < observations.size(); ++j) {
       double cos_theta = cos(particles[i].theta);
       double sin_theta = sin(particles[i].theta);
       double x_m = particles[i].x + cos_theta * observations[j].x - sin_theta * observations[j].y;
       double y_m = particles[i].y + sin_theta * observations[j].x + cos_theta * observations[j].y;
       transformed_observations.push_back(LandmarkObs{observations[j].id, x_m, y_m});
     }
     
     // Associate observation with landmarks
     dataAssociation(landmarks_within_range, transformed_observations);
     
     // Update steps
     // Reset the weight
     particles[i].weight = 1.0;
     
     // go through each observation and update the weight
     for (unsigned int j = 0; j < transformed_observations.size(); ++j) {
       
       int k_found;
       // find landmark within range by matching id with current observation
       for (unsigned int k = 0; k < landmarks_within_range.size(); ++k) {
         
         // matching ids are guaranteed to be found
         if (landmarks_within_range[k].id == transformed_observations[j].id) {
           k_found = k;
           break;
         }
       }
       
       double dx = transformed_observations[j].x - landmarks_within_range[k_found].x;
       double dy = transformed_observations[j].y - landmarks_within_range[k_found].y;
       
       // calculate praticle's final weight:
       // product of each measurement's Multivariate-Gaussian probability density
       double weight = exp(-dx * dx / (2 * std_landmark[0] * std_landmark[0]) -dy * dy / (2 * std_landmark[1] * std_landmark[1])) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
       if (weight == 0) {
         particles[i].weight *= 0.00001;
       } else {
         particles[i].weight *= weight;
       }
     } 
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<double> weights;
  double max_weight = std::numeric_limits<double>::min();
  // Create vector of weights concurrently updating maximal weight value
  for(int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }
  
  // Produce discrete distributions
  std::uniform_real_distribution<double> dist_weights(0.0, max_weight);
  std::uniform_int_distribution<int> dist_particle_n(0, num_particles - 1);
  
  // Resampling wheel
  // while w[index] < beta:
  //   beta = beta - w[index]
  //   index = index + 1
  // select p[index]
  
  vector<Particle> resampled_particles;
  int current_particle_n = dist_particle_n(gen);
  double beta = 0.0;
    
  for (int i = 0; i < num_particles; ++i) {
    beta += dist_weights(gen) * 2.0;
    while(weights[current_particle_n] < beta) {
      beta -= weights[current_particle_n];
      current_particle_n = (current_particle_n + 1) % num_particles;
    }
    resampled_particles.push_back(particles[current_particle_n]);
  }
  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}