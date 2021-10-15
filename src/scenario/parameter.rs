use itertools::Itertools;
use std::io::prelude::*;
use std::{collections::HashMap, fs::File};
use toml::Value;

#[derive(Debug, Clone)]
pub enum Domain {
    Integer(i32, i32),
    Categorical(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub switch: String,
    pub domain: Domain,
}

impl Parameter {
    pub fn new(name: &str, switch: &str, domain: Domain) -> Self {
        Self {
            name: name.to_string(),
            switch: switch.to_string(),
            domain,
        }
    }

    pub(crate) fn domain_size(&self) -> usize {
        match &self.domain {
            Domain::Integer(min, max) => (*min..*max).len() + 1,
            Domain::Categorical(option) => option.len(),
        }
    }

    pub(crate) fn get_value(&self, idx: usize) -> String {
        match &self.domain {
            Domain::Integer(min, max) => (*min..=*max).collect_vec()[idx].to_string(),
            Domain::Categorical(option) => option[idx].clone(),
        }
    }
}

#[derive(Debug)]
pub struct Parameters {
    map_params: HashMap<String, Parameter>,
    pub(crate) params: Vec<Parameter>,
    nb_params: usize,
}

impl Parameters {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_file(parameter_file: Option<File>) -> Self {
        if parameter_file.is_none() {
            panic!("Cannot open parameter file")
        }

        let mut file = parameter_file.unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let parse = contents.parse::<Value>();

        let value = match &parse {
            Ok(value) => value.as_table(),
            Err(error) => panic!("{}", error),
        };

        let mut parameters: Parameters = Default::default();

        if let Some(toml_table) = value.clone() {
            for (_, value) in toml_table {
                let parameter = if value["type"].as_str().unwrap() == "i" {
                    let domain = value["domain"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_integer().unwrap() as i32)
                        .collect_vec();

                    Parameter::new(
                        value["name"].as_str().unwrap(),
                        value["switch"].as_str().unwrap(),
                        Domain::Integer(domain[0], domain[1]),
                    )
                } else {
                    let domain = value["domain"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_str().unwrap().to_string())
                        .collect_vec();

                    Parameter::new(
                        value["name"].as_str().unwrap(),
                        value["switch"].as_str().unwrap(),
                        Domain::Categorical(domain),
                    )
                };

                parameters.add_parameter(parameter)
            }
        } else {
            panic!("Bad format parameters")
        }

        parameters
    }

    pub fn add_parameter(&mut self, parameter: Parameter) {
        if self.map_params.contains_key(&parameter.name) {
            panic!("Repeated parameter {}", parameter.name);
        }

        self.map_params
            .insert(parameter.name.clone(), parameter.clone());
        self.params.push(parameter.clone());
        self.nb_params += 1;
    }

    pub(crate) fn nb_params(&self) -> usize {
        self.nb_params
    }

    pub(crate) fn get_parameter(&self, index: usize) -> &Parameter {
        &self.params[index]
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            map_params: HashMap::<String, Parameter>::default(),
            params: Vec::<Parameter>::default(),
            nb_params: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Parameters {
    type Item = &'a Parameter;
    type IntoIter = std::slice::Iter<'a, Parameter>;

    fn into_iter(self) -> std::slice::Iter<'a, Parameter> {
        self.params.iter()
    }
}
