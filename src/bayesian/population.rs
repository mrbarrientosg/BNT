use std::{cell::RefCell, process::Command, rc::Rc};

use crate::scenario::{self, parameter::Domain, scenario::Scenario};
use rand::{prelude::StdRng, thread_rng, Rng, RngCore, SeedableRng};
use rayon::{
    iter::{IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Debug, Clone)]
pub(crate) enum SolutionType {
    Integer(usize),
    Categorical(String),
}

#[derive(Debug, Clone)]
pub(crate) struct Individual {
    pub(crate) id: usize,
    pub(crate) solution: Vec<usize>,
    pub(crate) fitness: f64,
}

impl Individual {
    fn new(scenario: &Scenario) -> Self {
        let mut solution: Vec<usize> = vec![];
        let mut trng = thread_rng();

        for parameter in scenario.parameters() {
            let value = match parameter.domain {
                Domain::Integer(_, _) => trng.gen_range(0..parameter.domain_size()) as usize,
                Domain::Categorical(_) => trng.gen_range(0..parameter.domain_size()) as usize,
            };
            solution.push(value);
        }

        Self {
            id: 0,
            solution,
            fitness: -1.0,
        }
    }

    pub(crate) fn from_sample(sample: &Vec<usize>) -> Self {
        Self {
            id: 0,
            solution: sample.clone(),
            fitness: -1.0,
        }
    }

    pub(crate) fn get_index(&self, idx: usize) -> usize {
        self.solution[idx]
    }

    pub(crate) fn print_solution(&self, scenario: &Scenario) {
        for (parameter, &idx) in scenario.parameters().into_iter().zip(self.solution.iter()) {
            print!("{}={} ", parameter.name, parameter.get_value(idx));
        }
    }

    pub(crate) fn get_configuration(&self, scenario: &Scenario) -> Vec<SolutionType> {
        let mut configuration: Vec<SolutionType> = vec![];

        for (parameter, &idx) in scenario.parameters().into_iter().zip(self.solution.iter()) {
            match parameter.domain {
                Domain::Integer(_, _) => configuration.push(SolutionType::Integer(
                    parameter.get_value(idx).parse().unwrap(),
                )),
                Domain::Categorical(_) => {
                    configuration.push(SolutionType::Categorical(parameter.get_value(idx)))
                }
            }
        }

        configuration
    }

    pub(crate) fn run_target_runner(&mut self, scenario: &Scenario, seeds: &Vec<u32>) {
        for ((id, instance), seed) in scenario
            .train_instances()
            .iter()
            .enumerate()
            .zip(seeds.iter())
        {
            let mut command = Command::new(scenario.target_runner());

            command
                .arg(self.id.to_string())
                .arg(id.to_string())
                .arg(seed.to_string())
                .arg(instance);

            for (parameter, &idx) in scenario.parameters().into_iter().zip(self.solution.iter()) {
                command.arg(parameter.switch.clone() + &parameter.get_value(idx));
            }

            let result = command.output();

            match result {
                Ok(output) => {
                    let error = match String::from_utf8(output.stderr) {
                        Ok(err) => err,
                        Err(_) => "".to_string(),
                    };

                    if !error.is_empty() {
                        eprintln!("{}", error);
                    }

                    let fitness: f64 = String::from_utf8(output.stdout)
                        .expect("Bad output")
                        .trim()
                        .parse()
                        .unwrap();

                    self.fitness += fitness;
                }
                Err(error) => panic!("{}", error),
            }
        }

        self.fitness /= scenario.train_instances().len() as f64;
    }
}

#[derive(Debug)]
pub(crate) struct Population {
    individuals: Vec<Individual>,
    population_size: usize,
    select_size: usize,
    last_individual_id: usize,
}

impl Population {
    pub(crate) fn new(population_size: usize, select_size: usize) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            individuals: vec![],
            population_size,
            select_size,
            last_individual_id: 0,
        }))
    }

    pub(crate) fn initialize(&mut self, scenario: &Scenario) {
        for _ in 0..self.population_size {
            let mut individual = Individual::new(scenario);
            individual.id = self.last_individual_id;
            self.individuals.push(individual);
            self.last_individual_id += 1;
        }

        let rng = thread_rng();
        let mut r = StdRng::from_rng(rng.clone()).unwrap();
        let seeds: Vec<u32> = (0..scenario.train_instances().len())
            .map(|_| r.next_u32())
            .collect();

        self.individuals
            .par_iter_mut()
            .for_each(|indi| indi.run_target_runner(scenario, &seeds));
    }

    pub(crate) fn population_size(&self) -> usize {
        self.population_size
    }

    pub(crate) fn run_new_individuals(
        &mut self,
        samples: &Vec<Vec<usize>>,
        scenario: &Scenario,
    ) -> Vec<Individual> {
        let mut individuals: Vec<Individual> = vec![];

        for sample in samples.iter() {
            let mut new_individual = Individual::from_sample(sample);
            new_individual.id = self.last_individual_id;
            individuals.push(new_individual);
            self.last_individual_id += 1;
        }

        let rng = thread_rng();

        let mut r = StdRng::from_rng(rng.clone()).unwrap();
        let seeds: Vec<u32> = (0..scenario.train_instances().len())
            .map(|_| r.next_u32())
            .collect();

        individuals
            .par_iter_mut()
            .for_each(|indi| indi.run_target_runner(scenario, &seeds));

        let copy_individuals = individuals.clone();

        self.individuals.append(&mut individuals);
        self.population_size += individuals.len();

        copy_individuals
    }

    pub(crate) fn best(&self) -> Individual {
        self.individuals[0].clone()
    }

    pub(crate) fn reduce(&mut self) {
        for _ in 0..(self.population_size - self.select_size) {
            self.individuals.pop();
        }

        self.population_size = self.select_size;
    }

    pub(crate) fn sort(&mut self) {
        self.individuals
            .par_sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    }
}

impl<'a> IntoIterator for &'a Population {
    type Item = &'a Individual;
    type IntoIter = std::slice::Iter<'a, Individual>;

    fn into_iter(self) -> std::slice::Iter<'a, Individual> {
        self.individuals.iter()
    }
}
