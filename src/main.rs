use crate::{
    bayesian_tuning::{BayesianConfig, BayesianTuning},
    scenario::scenario::Scenario,
};
use clap::{App, Arg};
use std::fs::File;

mod bayesian;
mod bayesian_tuning;
mod scenario;
mod utils;

fn main() {
    let matches = App::new("Bayesian Network Tunning")
        .version("1.0.0-alpha1")
        .author("Matias Barrientos")
        .about("Automatic Parameter Configuration")
        .arg(
            Arg::with_name("scenario")
                .long("scenario")
                .value_name("FILE")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("parameters")
                .long("parameters")
                .value_name("FILE")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max_iterations")
                .long("iterations")
                .short("i")
                .takes_value(true),
        )
        .get_matches();

    let mut scenario_file: Option<File> = None;
    if let Some(scenario_path) = matches.value_of("scenario") {
        match File::open(scenario_path) {
            Ok(file) => scenario_file = Some(file),
            Err(_) => scenario_file = None,
        }
    }

    let mut parameter_file: Option<File> = None;
    if let Some(parameter_path) = matches.value_of("parameters") {
        match File::open(parameter_path) {
            Ok(file) => parameter_file = Some(file),
            Err(_) => parameter_file = None,
        }
    }

    let scenario = Scenario::from_file(scenario_file, parameter_file);

    let mut config = BayesianConfig::new(&scenario);

    if let Some(iterations) = matches.value_of("max_iterations") {
        match iterations.parse::<usize>() {
            Ok(iter) => config.max_iterations = iter,
            Err(_) => (),
        }
    }

    let mut tuning = BayesianTuning::new(&scenario, config);

    tuning.run();
}

// let param_domain_size: usize = self
// .scenario
// .parameters()
// .get_parameter(current_node.index())
// .domain_size();

// let mut parents: Vec<usize> = vec![current_node.index()];

// self.graph
// .parents(*current_node)
// .iter(&self.graph)
// .for_each(|(_, parent)| {
//     parents.push(parent.index());
// });

// let counts: Vec<usize> = vec![];
// //self.alpha(*current_node,  &parents);
// let mut index = sample[parents[0]];

// for i in (parents.len() - 1)..=0 {
// let mut value: usize = 1;
// for _ in parents.len()..i {
//     value *= self
//         .scenario
//         .parameters()
//         .get_parameter(parents[i])
//         .domain_size();
// }
// index += sample[parents[i]] * value;
// }

// let mut il: Vec<usize> = vec![0; param_domain_size];

// let mut value: usize = 1;
// for i in 0..parents.len() {
// if parents[i] != current_node.index() {
//     value *= self
//         .scenario
//         .parameters()
//         .get_parameter(parents[i])
//         .domain_size();
// }
// }

// for i in 0..param_domain_size {
// il[i] = index + (i * value);
// }

// let mut a: Vec<f64> = vec![0.0; param_domain_size];
// let mut sum: f64 = 0.0;

// for i in 0..param_domain_size {
// a[i] = counts[il[i]] as f64;
// sum += a[i];
// }

// let mut prob: Vec<f64> = vec![0.0; param_domain_size];
// for i in 0..param_domain_size {
// prob[i] += a[i] / sum;
// }

// let mut ct: f64 = 0.0;

// for i in 0..param_domain_size {
// if prob[i] <= 0.0 {
//     ct += 1.0;
// }
// }

// ct *= 0.001 / (param_domain_size as f64 - ct);

// for i in 0..param_domain_size {
// if prob[i] > 0.0 {
//     prob[i] -= ct;
// } else {
//     prob[i] = 0.001;
// }
// }

// prob

// fn main() {
//     let it = (0..2).permutations(2);
//     for (a) in it.into_iter() {
//         println!("{:?}", a);
//     }

//     // let mut parents = vec![];

//     // self.graph
//     //     .parents(current_node)
//     //     .iter(&self.graph)
//     //     .for_each(|(_, parent)| {
//     //         parents.push(parent.index());
//     //     });

//     // // parents.push(candidate_node.index());

//     // let r_i = self
//     //     .scenario
//     //     .parameters()
//     //     .get_parameter(current_node.index())
//     //     .domain_size();

//     // let q_i: usize = if parents.len() > 0 {
//     //     parents
//     //         .iter()
//     //         .map(|i| {
//     //             if *i != current_node.index() {
//     //                 self.scenario.parameters().get_parameter(*i).domain_size()
//     //             } else {
//     //                 1
//     //             }
//     //         })
//     //         .product()
//     // } else {
//     //     0
//     // };

//     // let alpha = self.alpha(current_node, &parents);

//     // if q_i > 0 {
//     //     let mut nij: Vec<usize> = vec![0; q_i];

//     //     for j in 0..q_i {
//     //         for i in 0..r_i {
//     //             nij[j] += alpha[(r_i * j + i)];
//     //         }
//     //     }

//     //     for j in 0..q_i {
//     //         prod *= (r_i - 1).checked_factorial().unwrap_or(1) as f64
//     //             / (nij[j] + r_i - 1).checked_factorial().unwrap_or(1) as f64;

//     //         for i in 0..r_i {
//     //             prod *= alpha[(r_i * j + i)].checked_factorial().unwrap_or(1) as f64;
//     //         }
//     //     }
//     // } else {
//     //     let nij: usize = alpha.iter().sum();

//     //     prod *= (r_i - 1).checked_factorial().unwrap_or(1) as f64
//     //         / (nij + r_i - 1).checked_factorial().unwrap_or(1) as f64;

//     //     for i in 0..r_i {
//     //         prod *= alpha[i].checked_factorial().unwrap_or(1) as f64;
//     //     }
//     // }

// }

// let has_parents = parents.len() > 0;

// let mut size: usize = parents
//     .iter()
//     .map(|i| self.scenario.parameters().get_parameter(*i).domain_size())
//     .product();

// size *= self
//     .scenario
//     .parameters()
//     .get_parameter(current_node.index())
//     .domain_size();

// let mut alpha: Vec<usize> = vec![0; size];

// if has_parents {
//     let parents_size = parents.len() - 1;
//     let last_parent = *parents.last().unwrap();

//     for individual in self.population.borrow().into_iter() {
//         let mut index: usize = 0;

//         if let SolutionType::Integer(value) = individual.solution[last_parent] {
//             index += value;
//         }

//         if parents_size > 0 {
//             for i in (parents_size - 1)..=0 {
//                 let parent = parents[i];
//                 let mut offset: usize = 1;
//                 for j in parents_size..i {
//                     offset *= self
//                         .scenario
//                         .parameters()
//                         .get_parameter(parents[j])
//                         .domain_size()
//                 }

//                 if let SolutionType::Integer(value) = individual.solution[parent] {
//                     index += value * offset;
//                 }
//             }
//         }

//         alpha[index] += 1;
//     }
// } else {
//     let values = group_by(
//         &self
//             .population
//             .borrow()
//             .into_iter()
//             .map(|a| a.get_value(current_node.index()))
//             .collect(),
//     );

//     values.iter().for_each(|(idx, value)| {
//         alpha[*idx] = *value;
//     });
// }

// println!("{:?}", parents);
// println!("{:?}", alpha);
// println!();

// alpha
