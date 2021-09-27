use crate::{
    bayesian::{
        bayesian::BayesianNetwork,
        population::{Individual, Population},
    },
    scenario::scenario::Scenario,
};

use daggy::petgraph::dot::{Config, Dot};
use is_executable::IsExecutable;
use rand::{RngCore, SeedableRng, prelude::StdRng, thread_rng};
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
}

impl<'a> BayesianTuning<'a> {
    pub fn new(scenario: &'a Scenario, config: BayesianConfig) -> Self {
        let population = Population::new(scenario, config.population_size, config.select_size);

        let network = BayesianNetwork::new(scenario, Rc::clone(&population));

        Self {
            scenario,
            network,
            population,
            config,
        }
    }

    pub fn run(&mut self) {
        let target_runner_path = Path::new(self.scenario.target_runner());

        if !target_runner_path.is_executable() {
            panic!("Target runner is not an executable file.");
        }

        self.population.try_borrow_mut().unwrap().sort();

        let mut best: Option<Individual> = Some(self.population.try_borrow().unwrap().best());
        let rng = thread_rng();

        for i in 0..self.config.max_iterations {
            self.population.try_borrow_mut().unwrap().reduce();

            self.network.construct_network();

            let samples = self.network.sample(self.config.nb_children);

            let mut individuals: Vec<Individual> = vec![];

            for sample in samples.iter() {
                individuals.push(Individual::from_sample(sample));
            }

            {
                let scenario = self.scenario;
                let mut r = StdRng::from_rng(rng.clone()).unwrap();
                let seeds: Vec<u32> = (0..scenario.train_instances().len()).map(|_| r.next_u32()).collect();

                individuals
                    .par_iter_mut()
                    .for_each(|indi| indi.run_target_runner(scenario, &seeds));
            }

            self.population.try_borrow_mut().unwrap().add_individuals(&mut individuals);
            self.population.try_borrow_mut().unwrap().sort();
            self.population.try_borrow_mut().unwrap().reduce();

            best = Some(self.population.try_borrow().unwrap().best());
            print!("iteration = {}, fitness = {} ", i, best.as_ref().unwrap().fitness);
            best.as_ref().unwrap().print_solution(self.scenario);
        }
        println!(
            "{:?}",
            Dot::with_config(self.network.dag(), &[Config::EdgeNoLabel])
        );
        println!("Best:");
        best.as_ref().unwrap().print_solution(self.scenario);
    }
}