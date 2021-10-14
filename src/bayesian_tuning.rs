use crate::{
    bayesian::{
        bayesian::BayesianNetwork,
        population::{Individual, Population, SolutionType},
    },
    scenario::scenario::Scenario,
};

use daggy::petgraph::dot::{Config, Dot};
use is_executable::IsExecutable;
use itertools::Itertools;
use rand::{prelude::StdRng, thread_rng, RngCore, SeedableRng};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::{cell::RefCell, path::Path, rc::Rc};

pub struct BayesianConfig {
    pub population_size: usize,
    pub max_iterations: usize,
    pub select_size: usize,
    pub nb_children: usize,
}

impl BayesianConfig {
    pub fn new(scenario: &Scenario) -> Self {
        let population_size = 10 * scenario.parameters().nb_params();
        Self {
            population_size,
            max_iterations: 5 * scenario.parameters().nb_params(),
            select_size: (0.3 * population_size as f64).ceil() as usize,
            nb_children: population_size / 2,
        }
    }
}
pub struct BayesianTuning<'a> {
    scenario: &'a Scenario,
    population: Rc<RefCell<Population>>,
    network: BayesianNetwork<'a>,
    config: BayesianConfig,
    configurations: Vec<Individual>,
}

impl<'a> BayesianTuning<'a> {
    pub fn new(scenario: &'a Scenario, config: BayesianConfig) -> Self {
        let population = Population::new(config.population_size, config.select_size);

        let network = BayesianNetwork::new(scenario, Rc::clone(&population));

        Self {
            scenario,
            network,
            population,
            config,
            configurations: vec![],
        }
    }

    pub(crate) fn save_configurations(&self) {
        let mut csv = csv::Writer::from_path("configurations.csv").unwrap();

        let mut columns = self
            .scenario
            .parameters()
            .into_iter()
            .map(|param| param.name.clone())
            .collect_vec();

        columns.insert(0, "id".to_string());
        columns.push("fitness".to_string());

        csv.write_record(columns).unwrap();

        for individual in &self.configurations {
            let mut values = individual
                .get_configuration(self.scenario)
                .into_iter()
                .map(|solution| match solution {
                    SolutionType::Integer(value) => value.to_string(),
                    SolutionType::Categorical(value) => value.clone(),
                })
                .collect_vec();

            values.insert(0, individual.id.to_string());
            values.push(individual.fitness.to_string());

            csv.write_record(values).unwrap();
        }

        csv.flush().unwrap();
    }

    pub fn run(&mut self) {
        let target_runner_path = Path::new(self.scenario.target_runner());

        if !target_runner_path.is_executable() {
            panic!("Target runner is not an executable file.");
        }

        println!("# Bayesian Network Tuning Parameter ----------");
        println!("# Version: 1.0.0");
        println!("# Author: Matias Barrientos");
        println!("# --------------------------------------------");
        println!("# Scenario Initialization");
        println!("# Population size: {}", self.config.population_size);
        println!("# Select individual size: {}", self.config.select_size);
        println!("# Iterations: {}", self.config.max_iterations);
        println!("# New individual size: {}", self.config.nb_children);
        println!("");

        self.population.try_borrow_mut().unwrap().initialize(self.scenario);
        self.population.try_borrow_mut().unwrap().sort();

        for individual in self.population.try_borrow().unwrap().into_iter() {
            self.configurations.push(individual.clone());
        }

        self.population.try_borrow_mut().unwrap().reduce();

        let mut best: Option<Individual> = Some(self.population.try_borrow().unwrap().best());
        let rng = thread_rng();

        for i in 0..self.config.max_iterations {
            println!("# Iteration {} of {}", i + 1, self.config.max_iterations);

            self.network.construct_network();

            let samples = self.network.sample(self.config.nb_children);

            let mut individuals: Vec<Individual> = vec![];

            for sample in samples.iter() {
                let new_individual = Individual::from_sample(sample);
                individuals.push(new_individual.clone());
                self.configurations.push(new_individual.clone());
            }

            {
                let scenario = self.scenario;
                let mut r = StdRng::from_rng(rng.clone()).unwrap();
                let seeds: Vec<u32> = (0..scenario.train_instances().len())
                    .map(|_| r.next_u32())
                    .collect();

                individuals
                    .par_iter_mut()
                    .for_each(|indi| indi.run_target_runner(scenario, &seeds));
            }

            self.population
                .try_borrow_mut()
                .unwrap()
                .add_individuals(&mut individuals);
            self.population.try_borrow_mut().unwrap().sort();
            self.population.try_borrow_mut().unwrap().reduce();

            best = Some(self.population.try_borrow().unwrap().best());

            println!(
                "Best-so-far configuration: {} \t fitness: {}",
                best.as_ref().unwrap().id,
                best.as_ref().unwrap().fitness
            );
            println!("Description of the best-so-far configuration:");
            best.as_ref().unwrap().print_solution(self.scenario);
            println!("");
            println!("Bayesian network representation:");
            println!(
                "{:?}",
                Dot::with_config(self.network.dag(), &[Config::EdgeNoLabel])
            );
            println!("");
        }

        self.save_configurations();

        println!(
            "Best-so-far configuration: {} \t fitness: {}",
            best.as_ref().unwrap().id,
            best.as_ref().unwrap().fitness
        );
        println!("Description of the best-so-far configuration:");
        best.as_ref().unwrap().print_solution(self.scenario);
        println!("");
        println!("Bayesian network representation:");
        println!(
            "{:?}",
            Dot::with_config(self.network.dag(), &[Config::EdgeNoLabel])
        );
    }
}
