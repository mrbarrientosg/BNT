use super::parameter::{Parameter, Parameters};
use itertools::Itertools;
use rayon::slice::ParallelSliceMut;
use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::io::prelude::*;

#[derive(Deserialize)]
pub struct Scenario {
    #[serde(skip)]
    parameters: Parameters,

    #[serde(default)]
    train_instances_dir: String,

    #[serde(default)]
    train_instances: Vec<String>,

    #[serde(default)]
    test_instances: Vec<String>,
    target_runner: String,
}

impl Scenario {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_file(scenario_file: Option<File>, parameter_file: Option<File>) -> Self {
        if scenario_file.is_none() {
            panic!("Cannot open scenario file")
        }

        let mut file = scenario_file.unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let mut scenario: Scenario = match toml::from_str(&contents) {
            Ok(value) => value,
            Err(error) => panic!("{}", error),
        };

        scenario.parameters = Parameters::from_file(parameter_file);

        if let Ok(dir) = fs::read_dir(scenario.train_instances_dir()) {
            let mut instances = dir
                .into_iter()
                .map(|file| {
                    if let Ok(file) = file {
                        if let Ok(path) = fs::canonicalize(file.path()) {
                            let str_path = path.clone().to_str().unwrap().to_string();
                            return Some(str_path.clone());
                        }
                    }
                    None
                })
                .filter(|result| result.is_some())
                .map(|path| path.unwrap())
                .collect_vec();

            instances.par_sort();

            scenario = scenario.add_train_instances(instances.iter().map(|i| i.as_str()).collect_vec());
        }

        scenario
    }

    #[inline]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    #[inline]
    pub fn train_instances(&self) -> &Vec<String> {
        &self.train_instances
    }

    #[inline]
    pub fn test_instances(&self) -> &Vec<String> {
        &self.test_instances
    }

    #[inline]
    pub fn target_runner(&self) -> &String {
        &self.target_runner
    }

    #[inline]
    pub fn train_instances_dir(&self) -> &String {
        &self.train_instances_dir
    }

    pub fn add_parameter(mut self, parameter: Parameter) -> Self {
        self.parameters.add_parameter(parameter);
        self
    }

    pub fn add_parameters(mut self, parameters: Vec<Parameter>) -> Self {
        for parameter in parameters {
            self.parameters.add_parameter(parameter);
        }
        self
    }

    pub fn add_train_instances(mut self, instances: Vec<&str>) -> Self {
        for instance in instances {
            self.train_instances.push(instance.to_string());
        }
        self
    }

    pub fn add_target_runner(mut self, target_runner: &str) -> Self {
        self.target_runner = target_runner.to_string();
        self
    }
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            parameters: Parameters::default(),
            train_instances_dir: Default::default(),
            train_instances: vec![],
            test_instances: vec![],
            target_runner: String::default(),
        }
    }
}
