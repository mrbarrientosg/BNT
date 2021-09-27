use std::{cell::RefCell, process::Command, rc::Rc};

use crate::scenario::{parameter::Domain, scenario::Scenario};
use rand::{Rng, RngCore, SeedableRng, prelude::StdRng, thread_rng};
use rayon::{
    iter::{IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

// #[derive(Debug, Clone)]
// pub(crate) enum SolutionType {
//     // TODO: Cambiar a que guarde el indice del rango, ver que hacer con los real despues
//     Integer(usize),
// }

#[derive(Debug, Clone)]
pub(crate) struct Individual {
    pub(crate) solution: Vec<usize>,
    pub(crate) fitness: f64,
}

impl Individual {
    fn new(scenario: &Scenario) -> Self {
        let mut solution: Vec<usize> = vec![];
        let mut rng = thread_rng();

        for parameter in scenario.parameters() {
            let value = match parameter.domain {
                Domain::Integer(_, _) => rng.gen_range(0..parameter.domain_size()) as usize,
                Domain::Categorical(_) => rng.gen_range(0..parameter.domain_size()) as usize,
            };
            solution.push(value);
        }

        Self {
            solution,
            fitness: -1.0,
        }
    }

    pub(crate) fn from_sample(sample: &Vec<usize>) -> Self {
        Self {
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
        println!()
    }
    
    pub(crate) fn run_target_runner(&mut self, scenario: &Scenario, seeds: &Vec<u32>) {    
        for (instance, seed) in scenario.train_instances().iter().zip(seeds.iter()) {
            let mut command = Command::new(scenario.target_runner());

            command.arg("-i").arg(instance).arg("-s").arg(seed.to_string());

            for (parameter, &idx) in scenario.parameters().into_iter().zip(self.solution.iter()) {
                command
                    .arg(parameter.switch.clone())
                    .arg(parameter.get_value(idx));
            }

            let result = command.output();

            match result {
                Ok(output) => {
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
}

impl Population {
    pub(crate) fn new(
        scenario: &Scenario,
        population_size: usize,
        select_size: usize,
    ) -> Rc<RefCell<Self>> {
        let mut individuals: Vec<Individual> = vec![];

        for _ in 0..population_size {
            individuals.push(Individual::new(scenario));
        }

        let rng = thread_rng();
        let mut r = StdRng::from_rng(rng.clone()).unwrap();
        let seeds: Vec<u32> = (0..scenario.train_instances().len()).map(|_| r.next_u32()).collect();

        individuals
            .par_iter_mut()
            .for_each(|indi| indi.run_target_runner(scenario, &seeds));

        Rc::new(RefCell::new(Self {
            individuals,
            population_size,
            select_size,
        }))
    }

    pub(crate) fn population_size(&self) -> usize {
        self.population_size
    }

    pub(crate) fn add_individuals(&mut self, individuals: &mut Vec<Individual>) {
        self.individuals.append(individuals);
        self.population_size += 1;
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
